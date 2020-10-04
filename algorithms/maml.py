import megengine as meg
import megengine.module as M
from megengine.module.module import _expand_structure, _is_parameter
from typing import Optional, List, Dict, Set, Callable, Any, Tuple


def _is_trainable_parameter(obj) -> bool:
  return _is_parameter(obj) and (obj.requires_grad == True)


def extract_paramters(model: M.Module, recursive: bool = True,
                      prefix: Optional[str] = None, predicate: Callable[[Any], bool] = lambda _: True,
                      seen: Optional[Set[int]] = None,
                      module_table: Optional[Dict[str, M.Module]] = None,
                      named_parameters: Optional[Dict[str, meg.Parameter]] = None
                      ) -> Tuple[Dict[str, M.Module], Dict[str, meg.Parameter]]:

  if seen is None:
    seen = set([id(model)])

  if module_table is None:
    module_table = {}

  if named_parameters is None:
    named_parameters = {}

  module_dict = vars(model)
  _prefix = "" if prefix is None else prefix + "."

  for key in sorted(module_dict):
    for expanded_key, leaf in _expand_structure(key, module_dict[key]):
      leaf_id = id(leaf)
      if leaf_id in seen:
        continue
      seen.add(leaf_id)

      if predicate(leaf):
        module_name = _prefix.rstrip('.')
        module_table[module_name] = model
        named_parameters[_prefix + expanded_key] = leaf

      if recursive and isinstance(leaf, M.Module):
        extract_paramters(leaf,
                          recursive=recursive,
                          prefix=_prefix + expanded_key,
                          predicate=predicate,
                          seen=seen,
                          module_table=module_table,
                          named_parameters=named_parameters)
  return module_table, named_parameters


def replace_parameter(module_table: Dict[str, M.Module],
                      named_updates: Dict[str, meg.Parameter]):
  for key, value in named_updates.items():
    module_name, param_name = key.rsplit('.', 1)
    exec(f"module_table['{module_name}'].{param_name}=value")


class MAML(object):
  def __init__(self, model: M.Module):
    super().__init__()
    self.model = model
    module_table, named_parameters = self.extract_trainable_paramters(self.model)
    self.module_table = module_table
    self.named_keys: List[str] = list(named_parameters.keys())
    self.trainable_params: List[meg.Parameter] = list(named_parameters.values())

  def replace_fast_parameter(self, updates: List[meg.Parameter]):
    replace_parameter(self.module_table, named_updates=dict(zip(self.named_keys, updates)))

  @staticmethod
  def extract_trainable_paramters(model: M.Module):
    return extract_paramters(model, predicate=_is_trainable_parameter)
