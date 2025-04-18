from dataclasses import asdict, dataclass, field
import json
import socket
from uuid import uuid4
from typing import ClassVar, Dict, Type

@dataclass
class Message:
    uuid: str = field(default_factory=lambda: str(uuid4()))
    sender: str = field(default_factory=socket.gethostname)

    # register subclasses
    _subclasses: ClassVar[Dict[str, Type["Message"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # use class name as key for subclasses
        cls._subclasses[cls.__name__] = cls

    def serialize(self) -> str:
        # use asdict to convert dataclass to dict
        data = asdict(self)
        data["__class__"] = self.__class__.__name__
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, json_str: str) -> "Message":
        """
        TODO: Invalid JSON handling
        """
        data = json.loads(json_str)
        class_name = data.pop("__class__", None)
        
        if class_name and class_name in cls._subclasses:
            return cls._subclasses[class_name](**data)
        else:
            return cls(**data)

@dataclass
class ErrorMessage(Message):
    reason: str = ""

@dataclass
class GPUErrorMessage(ErrorMessage):
    gpu_index: int = -1
    host_name: str = ""

@dataclass
class NodeErrorMessage(ErrorMessage):
    host_name: str = ""

if __name__ == "__main__":
    gpu_error = GPUErrorMessage(reason="GPU OOM", host_name="host1", gpu_index=0)
    serialized = gpu_error.serialize()
    print("Serialized:", serialized)

    deserialized = Message.deserialize(serialized)
    print("Deserialized:", deserialized)
    print("Is GPUErrorMessage?", isinstance(deserialized, GPUErrorMessage))