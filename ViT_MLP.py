from collections import OrderedDict

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Sequential

from transformer_utils import MultiHeadSelfAttention
from transformer_utils import MultilayerPerceptron, Tokenizer, ClassTokenConcatenator, PositionEmbeddingAdder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from basic_model import BasicModel


class TransformerBlock(Module):
	"""
	Transformer block
	"""
	def __init__(
		self,
		token_dim: int,
		multihead_attention_head_dim: int,
		multihead_attention_n_heads: int,
		multilayer_perceptron_hidden_dim: int,
		dropout_p: float,
		) -> None:
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each input token
			multihead_attention_head_dim (int): Dimension of the
			queries/keys/values per head for multi-head self-attention
			multihead_attention_n_heads (int): Number of heads for the multihead
			attention
			multilayer_perceptron_hidden_dim (int): Dimension of the hidden
			layer for the multilayer perceptron
			dropout_p (float): Probability for dropouts applied after the
			hidden layer and second linear layer of the multilayer perceptron
			and on the output of the multi-head self-attention
		"""
		super().__init__()

		self.layer_norm_1 = LayerNorm(
			normalized_shape=token_dim,
			)
		self.multi_head_attention = MultiHeadSelfAttention(
			token_dim=token_dim,
			head_dim=multihead_attention_head_dim,
			n_heads=multihead_attention_n_heads,
			dropout_p=dropout_p,
			)

		self.layer_norm_2 = LayerNorm(
			normalized_shape=token_dim,
			)
		self.multilayer_perceptron = MultilayerPerceptron(
			in_dim=token_dim,
			hidden_dim=multilayer_perceptron_hidden_dim,
			out_dim=token_dim,
			dropout_p=dropout_p,
			)

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the transformer blocks

		Args:
			input (Tensor): Input

		Returns (Tensor): Output of the transformer block
		"""
		residual = input
		output = self.layer_norm_1(input)
		output = self.multi_head_attention(output)
		output = output+residual

		residual = output
		output = self.layer_norm_2(output)
		output = self.multilayer_perceptron(output)
		output = output+residual

		return output


class Transformer(Module):
	"""
	Transformer
	"""
	def __init__(
		self,
		n_layers: int,
		token_dim: int,
		multihead_attention_head_dim: int,
		multihead_attention_n_heads: int,
		multilayer_perceptron_hidden_dim: int,
		dropout_p: float,
		) -> None:
		"""
		Sets up the modules

		Args:
			n_layers (int): Depth of the transformer
			token_dim (int): Dimension of each input token
			multihead_attention_head_dim (int): Dimension of the
			queries/keys/values per head for multi-head self-attention
			multihead_attention_n_heads (int): Number of heads for the multihead
			attention
			multilayer_perceptron_hidden_dim (int): Dimension of the hidden
			layer for the multilayer perceptron
			dropout_p (float): Probability for dropouts applied after the
			hidden layer and second linear layer of the multilayer perceptron
			and on the output of the multi-head self-attention
		"""
		super().__init__()

		transformer_blocks = []
		for i in range(1, n_layers+1):
			transformer_block = TransformerBlock(
				token_dim=token_dim,
				multihead_attention_head_dim=multihead_attention_head_dim,
				multihead_attention_n_heads=multihead_attention_n_heads,
				multilayer_perceptron_hidden_dim=multilayer_perceptron_hidden_dim,
				dropout_p=dropout_p,
				)

			transformer_block = (f'transformer_block_{i}', transformer_block)

			transformer_blocks.append(transformer_block)

		transformer_blocks = OrderedDict(transformer_blocks)
		self.transformer_blocks = Sequential(transformer_blocks)

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the transformer blocks sequentially

		Args:
			input (Tensor): Input

		Returns (Tensor): Output of the final transformer block
		"""
		output = self.transformer_blocks(input)
		return output


class VisionTransformer(Module):
	"""
	Vision transformer
	"""
	def __init__(
		self,
		token_dim: int,
		patch_size: int,
		image_size: int,
		n_layers: int,
		multihead_attention_head_dim: int,
		multihead_attention_n_heads: int,
		multilayer_perceptron_hidden_dim: int,
		dropout_p: float,
		n_classes: int,
		):
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each input token
			patch_size (int): Patch size for each token
			image_size (int): Image size
			n_layers (int): Depth of the transformer
			multihead_attention_head_dim (int): Dimension of the
			queries/keys/values per head for multi-head self-attention
			multihead_attention_n_heads (int): Number of heads for the multihead
			attention
			multilayer_perceptron_hidden_dim (int): Dimension of the hidden
			layer for the multilayer perceptron
			dropout_p (float): Probability for dropouts applied after the
			hidden layer and second linear layer of the multilayer perceptron
			and on the output of the multi-head self-attention
			n_classes (int): Number of classes
		"""
		super().__init__()

		self.tokenizer = Tokenizer(
			token_dim=token_dim,
			patch_size=patch_size,
			)

		self.class_token_concatenator = ClassTokenConcatenator(
			token_dim=token_dim,
			)

		n_tokens = (image_size//patch_size) ** 2
		n_tokens += 1
		self.position_embedding_adder = PositionEmbeddingAdder(
			n_tokens=n_tokens,
			token_dim=token_dim,
			)

		self.transformer = Transformer(
			n_layers=n_layers,
			token_dim=token_dim,
			multihead_attention_head_dim=multihead_attention_head_dim,
			multihead_attention_n_heads=multihead_attention_n_heads,
			multilayer_perceptron_hidden_dim=multilayer_perceptron_hidden_dim,
			dropout_p=dropout_p,
			)

		# self.head = Linear(
		# 	in_features=token_dim,
		# 	out_features=n_classes,
		# 	)
	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the vision transformer

		Args:
			input (Tensor): Input

		Returns (Tensor): Output of the vision transformer's head
		"""
		output = self.tokenizer(input)
		output = self.class_token_concatenator(output)
		output = self.position_embedding_adder(output)
		output = self.transformer(output)
		output = torch.sum(output, dim=1)   #Augment All model
		#output = output[:, -1]
		
		# output = self.head(output)
		return output

class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        #x = self.relu1(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

class ViT_MLP(BasicModel):
    def __init__(self, args):
        super(ViT_MLP, self).__init__(args)
        self.conv = VisionTransformer(
            token_dim=512, # Token dimension
            patch_size=5, # Patch size
            image_size=80, # Image size
            n_layers=8, # Depth of transformer
            multihead_attention_head_dim=512, # Dimension of query/key/value vectors
            multihead_attention_n_heads=8, # Number of heads for multi-head self-attention
            multilayer_perceptron_hidden_dim=2048, # The hidden dimension of the transformer's multilayer perceptrons
            dropout_p=0.1, # The rate of dropout
            n_classes=8,
        )
        self.mlp = mlp_module()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def compute_loss(self, output, target, _):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x):
        features = self.conv(x)
        score = self.mlp(features)
        return score, None
