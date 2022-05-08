from torch import cat, Tensor, zeros
from torch.nn import Conv2d, Dropout, GELU, Linear, Module, Parameter


class MultilayerPerceptron(Module):
	"""
	Multilayer perceptron with one hidden layer
	"""
	def __init__(
		self,
		in_dim: int,
		hidden_dim: int,
		out_dim: int,
		dropout_p: float,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Dimension of the input
			hidden_dim (int): Dimension of the hidden layer
			out_dim (int): Dimension of the output
			dropout_p (float): Probability for dropouts applied after the
			hidden layer and second linear layer
		"""
		super().__init__()

		self.lin_1 = Linear(
			in_features=in_dim,
			out_features=hidden_dim,
			)
		self.act_1 = GELU()
		self.dropout_1 = Dropout(p=dropout_p)
		self.lin_2 = Linear(
			in_features=hidden_dim,
			out_features=out_dim,
			)
		self.dropout_2 = Dropout(p=dropout_p)

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the multilayer perceptron

		Args:
			input (Tensor): Input

		Returns (Tensor): Output of the multilayer perceptron
		"""
		output = self.lin_1(input)
		output = self.act_1(output)
		output = self.dropout_1(output)
		output = self.lin_2(output)
		output = self.dropout_2(output)
		return output


class Tokenizer(Module):
	"""
	Tokenizes an image
	"""
	def __init__(
		self,
		token_dim: int,
		patch_size: int,
		) -> None:
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each token
			patch_size (int): Height/width of each patch
		"""
		super().__init__()

		self.input_to_tokens = Conv2d(
			in_channels=1,
			out_channels=token_dim,
			kernel_size=patch_size,
			stride=patch_size,
			)

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Tokenizes the input with patch embeddings

		Args:
			input (Tensor): Input

		Returns (Tensor): Tokens, in the shape of
		(batch_size, n_token, token_dim)
		"""
		output = self.input_to_tokens(input)
		output = output.flatten(start_dim=-2, end_dim=-1)
		output = output.transpose(-2, -1)
		return output


class ClassTokenConcatenator(Module):
	"""
	Concatenates a class token to a set of tokens
	"""
	def __init__(
		self,
		token_dim: int,
		):
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each token
		"""
		super().__init__()

		class_token = zeros(token_dim)
		self.class_token = Parameter(class_token)


	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Concatenates the class token to the input

		Args:
			input (Tensor): Input

		Returns (Tensor): The input, with the class token concatenated to
		it
		"""
		class_token = self.class_token.expand(len(input), 1, -1)
		output = cat((input, class_token), dim=1)
		return output


class PositionEmbeddingAdder(Module):
	"""
	Adds learnable parameters to tokens for position embedding
	"""
	def __init__(
		self,
		n_tokens: int,
		token_dim: int,
		):
		"""
		Sets up the modules

		Args:
			n_tokens (int): Number of tokens
			token_dim (int): Dimension of each token
		"""
		super().__init__()

		position_embedding = zeros(n_tokens, token_dim)
		self.position_embedding = Parameter(position_embedding)

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Adds the position embeddings to the input

		Args:
			input (Tensor): Input

		Returns (Tensor): The input, with the learnable parameters added
		"""
		output = input+self.position_embedding
		return output


from typing import Tuple

from torch import Tensor
from torch.nn import Dropout, Linear, Module
from torch.nn.functional import softmax


class QueriesKeysValuesExtractor(Module):
	"""
	Gets queries, keys, and values for multi-head self-attention
	"""
	def __init__(
		self,
		token_dim: int,
		head_dim: int,
		n_heads: int,
		) -> None:
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each input token
			head_dim (int): Dimension of the queries/keys/values per head
			n_heads (int): Number of heads
		"""
		super().__init__()

		self.head_dim = head_dim
		self.n_heads = n_heads

		queries_keys_values_dim = 3 * self.head_dim * self.n_heads
		self.input_to_queries_keys_values = Linear(
			in_features=token_dim,
			out_features=queries_keys_values_dim,
			bias=False,
			)

	def forward(
		self,
		input: Tensor,
		) -> Tuple[Tensor, Tensor, Tensor]:
		"""
		Gets queries, keys, and values from the input

		Args:
			input (Tensor): Input

		Returns (Tuple[Tensor, Tensor, Tensor]): Queries, keys, and values
		"""
		batch_size, n_tokens, token_dim = input.shape

		queries_keys_values = self.input_to_queries_keys_values(input)

		queries_keys_values = queries_keys_values.reshape(
			batch_size,
			3,
			self.n_heads,
			n_tokens,
			self.head_dim,
			)

		queries, keys, values = queries_keys_values.unbind(dim=1)
		return queries, keys, values


def get_attention(
	queries: Tensor,
	keys: Tensor,
	values: Tensor,
	) -> Tensor:
	"""
	Calculates attention

	Args:
		queries (Tensor): Queries
		keys (Tensor): Keys
		values (Tensor): Values

	Returns (Tensor): Attention calculated using the provided queries, keys,
	and values
	"""
	scale = queries.shape[-1] ** -0.5
	attention_scores = (queries @ keys.transpose(-2, -1)) * scale
	attention_probabilities = softmax(attention_scores, dim=-1)

	attention = attention_probabilities @ values
	return attention


class MultiHeadSelfAttention(Module):
	"""
	Multi-head self-attention
	"""
	def __init__(
		self,
		token_dim: int,
		head_dim: int,
		n_heads: int,
		dropout_p: float,
		) -> None:
		"""
		Sets up the modules

		Args:
			token_dim (int): Dimension of each input token
			head_dim (int): Dimension of the queries/keys/values per head
			n_heads (int): Number of heads
			dropout_p (float): Probability for dropout applied on the output
		"""
		super().__init__()

		self.query_keys_values_extractor = QueriesKeysValuesExtractor(
			token_dim=token_dim,
			head_dim=head_dim,
			n_heads=n_heads,
			)

		self.concatenated_heads_dim = n_heads*head_dim
		self.attention_to_output = Linear(
			in_features=self.concatenated_heads_dim,
			out_features=token_dim,
			)

		self.output_dropout = Dropout(
			p=dropout_p,
			)

	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Calculates attention from the input

		Args:
			input (Tensor): Input

		Returns (Tensor): Attention
		"""
		batch_size, n_tokens, token_dim = input.shape

		queries, keys, values = self.query_keys_values_extractor(input)

		attention = get_attention(
			queries=queries,
			keys=keys,
			values=values,
			)

		attention = attention.reshape(
			batch_size,
			n_tokens,
			self.concatenated_heads_dim,
			)

		output = self.attention_to_output(attention)
		output = self.output_dropout(output)
		return output
