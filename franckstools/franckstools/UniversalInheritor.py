#UniversalInheritor - Franck Rothen
from typing import Any
import torch
from pytorch_lightning import LightningModule

class UniversalInheritor:
    def __init__(self, parent):
        self.parent = parent

    # Pass all other method calls to the original parent
    def __getattr__(self, attr):
        if hasattr(self.parent, attr):
            return getattr(self.parent, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

class PlWrapper(LightningModule, UniversalInheritor):
    def __init__(self, parent):
        super().__init__()
        UniversalInheritor.__init__(self, parent=parent)
        if hasattr(parent, "wrapperID"):
            self.wrapperID = parent.wrapperID + 1
        else:
            self.wrapperID = 1
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.parent.forward(*args, **kwargs)
    
    def training_step(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        self.parent._current_fx_name = self._current_fx_name
        loss = self.parent.training_step(sample, batch_idx)
        return loss
    
    def validation_step(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        self.parent._current_fx_name = self._current_fx_name
        return self.parent.validation_step(sample, batch_idx)
    
    def predict_step(self, sample: tuple, _batch_idx: int) -> torch.Tensor:
        return self.parent.predict_step(sample, _batch_idx)
    
    def configure_optimizers(self) -> dict:
        return self.parent.configure_optimizers()
    
    def on_train_epoch_end(self) -> None:
        return self.parent.on_train_epoch_end()
    

def main():
    # Example usage:
    class ParentClass:
        def method1(self):
            print("Method 1")

        def method2(self):
            print("Method 2")

    parent_instance = ParentClass()


    #New class
    class ChildClass(UniversalInheritor):
        def __init__(self, parent):
            super().__init__(parent)
        
        def method1(self):
            self.parent.method1()  # Call the parent method
            print('Extension for Method 1')
        
    # Create an instance of UniversalInheritor that copies all methods from ParentClass
    child = ChildClass(parent_instance)

    # Call the extended method1
    child.method1()
    child.method2()

if __name__ == '__main__':
    main()
