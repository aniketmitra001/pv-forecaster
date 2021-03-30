import torch
from typing import Dict, List, Tuple
from pytorch_forecasting import TimeSeriesDataSet 
from pytorch_forecasting.models import BaseModel
from pytorch_forecasting.models.nn import MultiEmbedding
from pytorch_forecasting.models.base_model import BaseModelWithCovariates


class FullyConnectedModule(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int):
        super().__init__()

        # input layer
        module_list = [torch.nn.Linear(input_size, hidden_size), torch.nn.LeakyReLU()]
        # hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([torch.nn.Linear(hidden_size, hidden_size), torch.nn.LeakyReLU()])
        # output layer
        module_list.extend([torch.nn.Linear(hidden_size, output_size), torch.nn.Sigmoid()])

        self.sequential = torch.nn.Sequential(*module_list)

      

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # output of shape batch_size x n_timesteps_out
        return self.sequential(x)

class FullyConnectedModelWithCovariates(BaseModelWithCovariates):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        x_reals: List[str],
        x_categoricals: List[str],
        embedding_sizes: Dict[str, Tuple[int, int]],
        embedding_labels: Dict[str, List[str]],
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_categoricals_encoder: List[str],
        time_varying_categoricals_decoder: List[str],
        time_varying_reals_encoder: List[str],
        time_varying_reals_decoder: List[str],
        embedding_paddings: List[str],
        categorical_groups: Dict[str, List[str]],
        **kwargs,
    ):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)

        # create embedder - can be fed with x["encoder_cat"] or x["decoder_cat"] and will return
        # dictionary of category names mapped to embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # calculate the size of all concatenated embeddings + continous variables
        n_features = sum(
            embedding_size for classes_size, embedding_size in self.hparams.embedding_sizes.values()
        ) + len(self.reals)

        # create network that will be fed with continious variables and embeddings
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size * n_features,
            output_size=self.hparams.output_size,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )
        

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        batch_size = x["encoder_lengths"].size(0)
        embeddings = self.input_embeddings(x["encoder_cat"])  # returns dictionary with embedding tensors
        network_input = torch.cat(
            [x["encoder_cont"]]
            + [
                emb
                for name, emb in embeddings.items()
                if name in self.encoder_variables or name in self.static_variables
            ],
            dim=-1,
        )
        prediction = self.network(network_input.view(batch_size, -1))
        
        # We need to return a dictionary that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        return dict(prediction=prediction, target_scale=x["target_scale"])

    
    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"

        return super().from_dataset(dataset, **new_kwargs)

    @classmethod
    def optimize(cls,  dataset: TimeSeriesDataSet, num_steps: int,  **kwargs):
        
        model = FullyConnectedModelWithCovariates.from_dataset(dataset, **kwargs)
        dataloader = dataset.to_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), model.hparams.learning_rate )
        criteria = torch.nn.L1Loss() 
        for step in range(num_steps):
            optimizer.zero_grad()
            x_train, y_train = next(iter(dataloader))
            # Forward pass
            y_pred = model(x_train)['prediction']
            # Compute Loss
            loss = criteria(y_pred, y_train[0])

            print('Step {}: train loss: {}'.format(step, loss.item()))

            # Backward pass
            loss.backward()
            optimizer.step()

        return model    

   





