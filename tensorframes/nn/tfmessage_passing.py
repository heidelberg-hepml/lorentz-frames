from typing import Any, Dict

from torch_geometric.nn import MessagePassing

from tensorframes.lframes import ChangeOfLFrames, LFrames


class TFMessagePassing(MessagePassing):
    """TFMessagePassing class represents a message passing algorithm in the tensorframes formalism.

    https://arxiv.org/abs/2405.15389v1
    """

    def __init__(self, params_dict: Dict[str, Dict[str, Any]], aggr="add") -> None:
        """Initializes a new instance of the TFMessagePassing class.

        Args:
            params_dict (Dict[str, Dict[str, Any]]): A dictionary containing the parameters for the message passing algorithm and the corresponding representations. Params which are not listed here are not transformed E.g.:
            {
                "feat_0": {
                    "type": "local"
                    "rep": TensorReps("1x0n")
                },
                "feat_1": {
                    "type": "global"
                    "rep": Irreps("1x0n")
                },
            }
        """
        self.params_dict = params_dict

        for key, value in self.params_dict.items():
            if value["type"] is not None:
                self.params_dict[key]["transform"] = value["rep"].get_transform_class()

        super().__init__(aggr=aggr)

        # Register hooks to call before propagating and before sending messages
        self.register_propagate_forward_pre_hook(self.pre_propagate_hook)
        self.register_message_forward_pre_hook(self.pre_message_hook)

    def pre_propagate_hook(self, module: Any, inputs: tuple) -> tuple:
        """A hook method called before propagating messages in the message passing algorithm. We
        save the lframes in the class variable and remove it from the inputs dictionary.

        Args:
            module (Any): The module object.
            inputs (tuple): The inputs dictionary.

        Returns:
            tuple: The modified inputs dictionary.
        """
        assert inputs[-1].get("lframes") is not None, "lframes are not in the propagate inputs"

        self._lframes = inputs[-1].pop("lframes")
        self._edge_index = inputs[0]

        return inputs

    def pre_message_hook(self, module: Any, inputs: tuple) -> tuple:
        """Pre-message hook method that is called before passing messages in the message passing
        algorithm. We transform the features according to the representations in the params_dict.

        Args:
            module (Any): The module object.
            inputs (tuple): The inputs dictionary.

        Returns:
            tuple: The modified inputs dictionary.
        """

        # calculate lframes_i, lframes_j and the U matrix
        if isinstance(self._lframes, tuple):
            lframes_i = self._lframes[1].index_select(self._edge_index[1])
            lframes_j = self._lframes[0].index_select(self._edge_index[0])
        elif isinstance(self._lframes, LFrames):
            lframes_i = self._lframes.index_select(self._edge_index[1])
            lframes_j = self._lframes.index_select(self._edge_index[0])
        else:
            raise ValueError(
                f"lframes should be either a tuple or an LFrames object but is {type(self._lframes)}"
            )

        U = ChangeOfLFrames(lframes_start=lframes_j, lframes_end=lframes_i)

        # now go through the params_dict and get the representations and transform the features in the right way
        for param, param_info in self.params_dict.items():
            if param_info["type"] == "local":
                assert param + "_j" in inputs[-1], f"Key {param}_j not in inputs"
                # transform the features according to the representation
                inputs[-1][param + "_j"] = param_info["transform"](inputs[-1][param + "_j"], U)
            elif param_info["type"] == "global":
                if inputs[-1].get(param) is not None:
                    inputs[-1][param] = param_info["transform"](inputs[-1][param], lframes_i)
                if inputs[-1].get(param + "_j") is not None:
                    inputs[-1][param + "_j"] = param_info["transform"](
                        inputs[-1][param + "_j"], lframes_i
                    )
                if inputs[-1].get(param + "_i") is not None:
                    inputs[-1][param + "_i"] = param_info["transform"](
                        inputs[-1][param + "_i"], lframes_i
                    )

        return inputs
