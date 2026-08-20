"""Modules custom pour une réimplémentation "best-effort" de BGLE-YOLO
(Zhang et al., "BGLE-YOLO: A Lightweight Model for Underwater
Bio-Detection", 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11902696/).

Reconstruction à partir de la description du papier, pas du code original
(non publié) : interprétation fidèle à l'esprit de chaque module (conv
multi-échelle, attention globale/locale, convolution à différences
reparamétrée), mais pas une reproduction byte-perfect — certains détails
(ratios de canaux exacts, nombre de répétitions, placement précis dans le
réseau) ne sont pas donnés dans le papier et ont été choisis de façon
raisonnable.

Les trois modules conservent le nombre de canaux (in == out) : ça permet de
les insérer dans un YAML de modèle ultralytics sans patcher `parse_model()`,
dont le calcul par défaut pour un module inconnu (`c2 = ch[f]`) devient
alors correct automatiquement. Les changements de résolution/canaux restent
gérés par les Conv/Concat/Upsample standard du YAML autour de ces blocs.

  - EMC   : Efficient Multi-Scale Convolution (backbone)
  - GLSA  : Global-to-Local Spatial Aggregation (composant du module BIG du
            papier, neck) -- la fusion BiFPN pondérée elle-même est gérée
            dans le YAML via des Conv 1x1 + additions pondérées standard,
            faute de détail suffisant dans le papier pour reproduire leur
            variante de BiFPN.
  - LSHDetect : tête de détection légère (DEConv + GroupNorm), sous-classe
            de ultralytics.nn.modules.head.Detect pour rester compatible
            avec la loss/décodage DFL existants -- seuls les blocs conv
            internes (cv2/cv3) sont remplacés.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, DWConv
from ultralytics.nn.modules.head import Detect


class EMC(nn.Module):
    """Efficient Multi-Scale Convolution (backbone, cf. docstring module).

    La moitié des canaux passe sans convolution (économie de calcul, même
    idée que GhostConv/PConv), l'autre moitié est splittée en deux chemins
    parallèles (noyaux 3x3 et 5x5) pour capter des échelles différentes,
    fusionnés par une conv 1x1 finale. Conserve le nombre de canaux.
    """

    def __init__(self, c1: int, e: float = 0.5):
        super().__init__()
        self.c_bypass = c1 // 2
        c_active = c1 - self.c_bypass
        self.c_a = c_active // 2
        self.c_b = c_active - self.c_a
        self.cv3 = Conv(self.c_a, self.c_a, k=3) if self.c_a > 0 else None
        self.cv5 = Conv(self.c_b, self.c_b, k=5) if self.c_b > 0 else None
        self.fuse = Conv(c1, c1, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bypass, active = x[:, : self.c_bypass], x[:, self.c_bypass :]
        x_a, x_b = active[:, : self.c_a], active[:, self.c_a :]
        y_a = self.cv3(x_a) if self.cv3 is not None else x_a
        y_b = self.cv5(x_b) if self.cv5 is not None else x_b
        return self.fuse(torch.cat([bypass, y_a, y_b], dim=1))


class GSA(nn.Module):
    """Global Spatial Attention : self-attention sur la carte de features
    aplatie, capture les relations longue-distance entre pixels."""

    def __init__(self, c: int):
        super().__init__()
        self.q = nn.Conv2d(c, c, 1)
        self.k = nn.Conv2d(c, c, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.scale = c**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).flatten(2)  # b,c,hw
        k = self.k(x).flatten(2)
        v = self.v(x).flatten(2)
        attn = torch.softmax(q.transpose(1, 2) @ k * self.scale, dim=-1)  # b,hw,hw
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return out


class LSA(nn.Module):
    """Local Spatial Attention : 3 convs 1x1 + une conv séparable 3x3
    depth-wise pour les features locales."""

    def __init__(self, c: int):
        super().__init__()
        self.cv1 = nn.Conv2d(c, c, 1)
        self.cv2 = nn.Conv2d(c, c, 1)
        self.dw = DWConv(c, c, k=3, act=False)
        self.cv3 = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.dw(self.cv2(self.cv1(x))))


class GLSA(nn.Module):
    """Global-to-Local Spatial Aggregation : combine GSA + LSA (composant
    du module BIG du papier). Conserve le nombre de canaux."""

    def __init__(self, c1: int):
        super().__init__()
        self.gsa = GSA(c1)
        self.lsa = LSA(c1)
        self.fuse = Conv(c1 * 2, c1, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.gsa(x), self.lsa(x)], dim=1))


class DEConv(nn.Module):
    """Detail-Enhanced Convolution : 5 noyaux dérivés d'un même poids de
    base (vanille + 4 convolutions de différence : centrale, angulaire,
    horizontale, verticale), sommés en un seul noyau équivalent
    (reparamétrisation -- un seul appel conv2d à l'inférence, pas de coût
    supplémentaire). Remplace les Conv standard dans la tête LSH."""

    def __init__(self, c1: int, c2: int, k: int = 3, theta: float = 0.2):
        super().__init__()
        self.k = k
        self.theta = theta  # atténuation des 4 termes de différence, cf. docstring _effective_kernel
        self.weight = nn.Parameter(torch.empty(c2, c1, k, k))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(c2))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def _effective_kernel(self) -> torch.Tensor:
        """Combine le poids de base et les 4 termes de différence en un seul
        noyau. Sommer les 5 variantes à pleine échelle sans atténuation rend
        le noyau effectif ~2.7x plus fort qu'une conv standard dès
        l'initialisation (la case centrale cumule trois termes de
        soustraction qui se recouvrent) -- empilé sur 2 DEConv par branche ×
        3 échelles, l'instabilité s'amplifie et fait diverger
        l'entraînement (pertes NaN). Chaque terme de différence est donc
        exprimé comme une perturbation localisée (nulle hors des positions
        qu'elle affecte) et atténué par `theta` avant d'être ajouté au poids
        de base, qui reste à l'échelle d'une conv standard."""
        w = self.weight
        center = self.k // 2

        delta_cdc = torch.zeros_like(w)
        delta_cdc[..., center, center] = -w.sum(dim=(-2, -1))

        delta_hdc = torch.zeros_like(w)
        delta_hdc[..., center, :] = -w[..., center, :].sum(dim=-1, keepdim=True).expand(-1, -1, self.k)

        delta_vdc = torch.zeros_like(w)
        delta_vdc[..., :, center] = -w[..., :, center].sum(dim=-1, keepdim=True).expand(-1, -1, self.k)

        delta_adc = w - w.flip(dims=(-2, -1))

        return w + self.theta * (delta_cdc + delta_hdc + delta_vdc + delta_adc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self._effective_kernel(), self.bias, padding=self.k // 2)
        return self.act(self.bn(y))


class LSHBranch(nn.Module):
    """Une branche (box ou cls) de la tête LSH : DEConv + GroupNorm au lieu
    de Conv+BatchNorm standard, suivi d'une projection finale 1x1."""

    def __init__(self, c1: int, c_mid: int, c_out: int, groups: int = 32):
        super().__init__()
        self.de1 = DEConv(c1, c_mid, k=3)
        self.gn1 = nn.GroupNorm(min(groups, c_mid), c_mid)
        self.de2 = DEConv(c_mid, c_mid, k=3)
        self.gn2 = nn.GroupNorm(min(groups, c_mid), c_mid)
        self.proj = nn.Conv2d(c_mid, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gn1(self.de1(x))
        x = self.gn2(self.de2(x))
        return self.proj(x)


class _AlignedSharedBox(nn.Module):
    """Wrapper autour d'une branche box partagée entre échelles : aligne
    d'abord les canaux de l'échelle (différents pour P3/P4/P5 dans une FPN
    standard) vers le canal commun de la branche partagée via une conv 1x1,
    puis appelle cette branche -- sans cet alignement, réutiliser le même
    LSHBranch à travers des échelles de canaux différents est impossible
    (mismatch de shape)."""

    def __init__(self, c_in: int, c_common: int, shared_branch: "LSHBranch"):
        super().__init__()
        self.align = nn.Conv2d(c_in, c_common, 1) if c_in != c_common else nn.Identity()
        self.shared = shared_branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(self.align(x))


class LSHDetect(Detect):
    """Tête de détection "Lightweight Shared Head" (LSH) : mêmes
    branches box (cv2) / classe (cv3) que Detect, mais construites avec
    DEConv+GroupNorm. "Partagée" (RPC, cf. papier) : la branche box (cv2)
    réutilise LE MÊME module DEConv+GroupNorm à travers toutes les
    échelles (poids partagés) ; la branche classe (cv3) garde des poids
    séparés par échelle (LPC), comme décrit dans le papier. Hérite
    forward()/décodage DFL de Detect sans changement (compatible avec la
    loss v8DetectionLoss existante d'ultralytics) -- bias_init() est
    réécrit ci-dessous car la version de Detect indexe cv2[i][-1]/cv3[i][-1]
    en supposant un nn.Sequential (comme dans Detect standard), alors que
    LSHBranch est un nn.Module avec un attribut .proj nommé, pas
    subscriptable de la même façon."""

    def __init__(self, nc: int = 80, ch: tuple = ()):
        super().__init__(nc, ch=ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        shared_box = LSHBranch(c2, c2, 4 * self.reg_max)
        self.cv2 = nn.ModuleList(_AlignedSharedBox(x, c2, shared_box) for x in ch)
        self.cv3 = nn.ModuleList(LSHBranch(x, c3, self.nc) for x in ch)

    def bias_init(self):
        """Équivalent de Detect.bias_init() adapté à LSHBranch.proj (pas de
        Sequential à indexer par [-1]) -- `a` est un _AlignedSharedBox ici,
        d'où `a.shared.proj` (la même instance partagée à chaque itération,
        donc les 3 assignations à la branche box sont redondantes mais sans
        effet de bord)."""
        import math

        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a.shared.proj.bias.data[:] = 2.0  # box
            b.proj.bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls
