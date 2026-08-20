"""Enregistre les modules custom (cf. bgle_modules.py) dans l'espace de
noms d'ultralytics.nn.tasks, pour que parse_model() (qui résout les noms
de modules d'un YAML via `globals()[m]`, cf. ultralytics/nn/tasks.py) les
trouve. Ultralytics ne fournit pas de mécanisme d'enregistrement officiel
pour des modules tiers -- c'est l'approche standard utilisée dans la
communauté pour des architectures custom sans forker le package.

EMC/GLSA sont à canaux préservés (in=out) : le fallback générique
d'ultralytics pour un module inconnu (`else: c2 = ch[f]`, cf.
parse_model()) est déjà correct pour eux, seuls leurs args YAML doivent
inclure c1 explicitement (pas d'auto-injection depuis `ch[f]` pour un
module hors de la liste `base_modules` d'ultralytics).

LSHDetect a plusieurs entrées ("from" = liste, une par échelle P3/P4/P5),
comme Detect -- mais parse_model() ne reconnaît Detect que par identité de
classe exacte (`m in frozenset({Detect, ...})`), pas par sous-classe, donc
LSHDetect tombe sinon dans le cas générique (`c2 = ch[f]`) qui plante sur
un "from" multi-index. Patché ci-dessous en régénérant parse_model() avec
une branche elif supplémentaire (insertion de texte + exec, pas de fork du
package) -- fragile en théorie face à un changement de version
d'ultralytics, mais la seule voie sans dupliquer toute la fonction (~200
lignes) ou modifier le package installé.

Usage :
    import src.nn.register  # noqa: F401 -- effet de bord, à importer avant YOLO(...)
"""
import inspect
import sys
import types

import ultralytics.nn.tasks as _tasks

from . import bgle_modules
from .bgle_modules import EMC, GLSA, LSHDetect

_tasks.EMC = EMC
_tasks.GLSA = GLSA
_tasks.LSHDetect = LSHDetect

# Compat pour les checkpoints entraînés avant que ce module ne soit importé
# de façon cohérente en "src.nn.*" : torch.load dépickle les classes custom
# sous le chemin de module qui était effectif au moment de l'entraînement
# ("nn.bgle_modules.*" pour les checkpoints antérieurs). Alias explicite
# plutôt que de dépendre d'un sys.path bricolé -- retirer une fois qu'aucun
# checkpoint existant ne référence plus "nn.*" (torch.load avertit sinon
# avec un ModuleNotFoundError).
sys.modules["nn"] = types.ModuleType("nn")
sys.modules["nn"].bgle_modules = bgle_modules
sys.modules["nn.bgle_modules"] = bgle_modules


def _patch_parse_model_for_lshdetect() -> None:
    anchor = "        elif m is Depth:"
    patch = (
        "        elif m is LSHDetect:\n"
        "            args = [args[0], [ch[x] for x in f]]\n"
        "            c2 = ch[f[0]]\n"
        + anchor
    )
    src = inspect.getsource(_tasks.parse_model)
    if anchor not in src:
        raise RuntimeError("register.py: ancre introuvable dans parse_model() -- version d'ultralytics incompatible avec ce patch.")
    if "LSHDetect" in src:
        return  # déjà patché (ex: réimport)
    patched_src = src.replace(anchor, patch, 1)
    exec(compile(patched_src, "<parse_model_patched_for_lshdetect>", "exec"), _tasks.__dict__)


_patch_parse_model_for_lshdetect()
