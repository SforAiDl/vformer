"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""


class Registry:
    """
    Class to register objects and then retrieve them by name.
    Parameters
    ----------
    name : str
        Name of the registry
    """

    def __init__(self, name):

        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):

        assert (
            name not in self._obj_map
        ), f"An object named '{name}' was already registered in '{self._name}' registry!"

        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        """
        Method to register an object in the registry
        Parameters
        ----------
        obj : object, optional
            Object to register, defaults to None (which will return the decorator)
        name : str, optional
            Name of the object to register, defaults to None (which will use the name of the object)
        """

        if obj is None:

            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        if name is None:  # pragma: no cover
            name = obj.__name__

        self._do_register(name, obj)  # pragma: no cover

    def get(self, name):
        """
        Method to retrieve an object from the registry
        Parameters
        ----------
        name : str
            Name of the object to retrieve
        Returns
        -------
        object
            Object registered under the given name
        """

        ret = self._obj_map.get(name)
        if ret is None:  # pragma: no cover
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
            )

        return ret

    def get_list(self):
        """
        Method to retrieve all objects from the registry
        Returns
        -------
        list
            List of all objects registered in the registry
        """

        return list(self._obj_map.keys())

    def __contains__(self, name):
        return name in self._obj_map  # pragma: no cover

    def __iter__(self):
        return iter(self._obj_map.items())  # pragma: no cover


ATTENTION_REGISTRY = Registry("ATTENTION")
DECODER_REGISTRY = Registry("DECODER")
ENCODER_REGISTRY = Registry("ENCODER")
MODEL_REGISTRY = Registry("MODEL")
