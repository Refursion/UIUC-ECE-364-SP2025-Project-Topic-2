class ProgressiveUnfreezer:
    def __init__(self, model, model_chosen="TinyBERT", mode="accuracy", patience=2, unfreeze_every=3):
        """
        mode: "accuracy" for based on valid acc, "epoch" for every N epochs
        """
        self.model = model
        self.mode = mode
        self.patience = patience
        self.unfreeze_every = unfreeze_every
        self.no_improve_epochs = 0
        self.best_valid_acc = 0.0
        self.current_unfreeze_idx = 0

        if model_chosen == "TinyBERT":
            self.layers_to_unfreeze = [
                "bert.encoder.layer.3",
                "bert.encoder.layer.2",
                "bert.encoder.layer.1",
                "bert.encoder.layer.0",
                "bert.embeddings"
            ]
        elif model_chosen == "ALBERT":
            self.layers_to_unfreeze = [
                "albert.encoder.albert_layer_groups.0.albert_layers.0",
                "albert.encoder",
                "albert.embeddings"
            ]
        else:
            raise ValueError(f"Unsupported model_chosen: {model_chosen}")

    def update(self, valid_accuracy=None, epoch=None):
        """
        Should be called every epoch.
        - In accuracy mode, pass valid_accuracy.
        - In epoch mode, pass epoch number.
        """
        if self.mode == "accuracy":
            if valid_accuracy is None:
                raise ValueError("In accuracy mode, must provide valid_accuracy to update().")
            if valid_accuracy > self.best_valid_acc:
                self.best_valid_acc = valid_accuracy
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1
            if self.no_improve_epochs >= self.patience:
                self._unfreeze_next_layer()
                self.no_improve_epochs = 0

        elif self.mode == "epoch":
            if epoch is None:
                raise ValueError("In epoch mode, must provide epoch number to update().")
            if (epoch + 1) % self.unfreeze_every == 0:
                self._unfreeze_next_layer()

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _unfreeze_next_layer(self):
        if self.current_unfreeze_idx >= len(self.layers_to_unfreeze):
            print(" All layers already unfrozen.")
            return
        target_layer = self.layers_to_unfreeze[self.current_unfreeze_idx]
        print(f" Unfreezing {target_layer}")
        for name, param in self.model.named_parameters():
            if target_layer in name:
                param.requires_grad = True
        self.current_unfreeze_idx += 1

    def freeze_all_except_classifier(self):
        """At start, freeze everything except classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
