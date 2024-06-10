from observatory.models.model_wrapper import ModelWrapper

from transformers import BertModel, BertTokenizer


class BERTModelWrapper(ModelWrapper):
    def get_model(self):
        try:
            model = BertModel.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            model = BertModel.from_pretrained(self.model_name)

        model = model.to(self.device)
        model.eval()

        return model

    def get_tokenizer(self):
        try:
            tokenizer = BertTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
        except OSError:
            tokenizer = BertTokenizer.from_pretrained(self.model_name)

        return tokenizer

    def get_max_input_size(self):
        return 512

    def get_column_embeddings(self):
        pass
