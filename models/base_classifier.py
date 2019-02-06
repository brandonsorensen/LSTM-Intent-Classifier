from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    @abstractmethod
    def _init_model(self):
        pass
    
    def _set_methods(self):
        try:
            self.model
        except AttributeError:
            raise Exception('No model attribute')
        self.predict_classes = self.model.predict_classes
        self.predict = self.model.predict
        self.fit = self.model.fit
        self.evaluate = self.model.evaluate
        self.compile = self.model.compile
        self.save = self.model.save
        self.summary = self.model.summary
