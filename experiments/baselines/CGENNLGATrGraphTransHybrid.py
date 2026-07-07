

import functools
import itertools
import math
import operator
import torch
from torch import nn

from lgatr import (
    LGATr,
    embed_vector,
    extract_scalar,
    get_num_spurions,
    get_spurions,
)

def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.
    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
        dim: int: starting dim, default: 0.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]

# Inspired by https://github.com/pygae/clifford
# copied from the itertools docs
def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )

class ShortLexBasisBladeOrder:
    def __init__(self, n_vectors):
        self.index_to_bitmap = torch.empty(2 ** n_vectors, dtype=int)
        self.grades = torch.empty(2 ** n_vectors, dtype=int)
        self.bitmap_to_index = torch.empty(2 ** n_vectors, dtype=int)
        for i, t in enumerate(_powerset([1 << i for i in range(n_vectors)])):
            bitmap = functools.reduce(operator.or_, t, 0)
            self.index_to_bitmap[i] = bitmap
            self.grades[i] = len(t)
            self.bitmap_to_index[bitmap] = i
            del t # enables an optimization inside itertools.combinations

def set_bit_indices(x: int):
    """Iterate over the indices of bits set to 1 in `x`, in ascending order"""
    n = 0
    while x > 0:
        if x & 1:
            yield n
        x = x >> 1
        n = n + 1

def count_set_bits(bitmap: int) -> int:
    """Counts the number of bits set to 1 in bitmap"""
    count = 0
    for i in set_bit_indices(bitmap):
        count += 1
    return count

def canonical_reordering_sign_euclidean(bitmap_a, bitmap_b):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    assuming a euclidean metric
    """
    a = bitmap_a >> 1
    sum_value = 0
    while a != 0:
        sum_value = sum_value + count_set_bits(a & bitmap_b)
        a = a >> 1
    if (sum_value & 1) == 0:
        return 1
    else:
        return -1

def canonical_reordering_sign(bitmap_a, bitmap_b, metric):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    given the supplied metric
    """
    bitmap = bitmap_a & bitmap_b
    output_sign = canonical_reordering_sign_euclidean(bitmap_a, bitmap_b)
    i = 0
    while bitmap != 0:
        if (bitmap & 1) != 0:
            output_sign *= metric[i]
        i = i + 1
        bitmap = bitmap >> 1
    return output_sign

def gmt_element(bitmap_a, bitmap_b, sig_array):
    """
    Element of the geometric multiplication table given blades a, b.
    The implementation used here is described in :cite:`ga4cs` chapter 19.
    """
    output_sign = canonical_reordering_sign(bitmap_a, bitmap_b, sig_array)
    output_bitmap = bitmap_a ^ bitmap_b
    return output_bitmap, output_sign

def construct_gmt(index_to_bitmap, bitmap_to_index, signature):
    n = len(index_to_bitmap)
    array_length = int(n * n)
    coords = torch.zeros((3, array_length), dtype=torch.int)
    k_list = coords[0, :]
    l_list = coords[1, :]
    m_list = coords[2, :]
    # use as small a type as possible to minimize type promotion
    mult_table_vals = torch.zeros(array_length)
    for i in range(n):
        bitmap_i = index_to_bitmap[i]
        for j in range(n):
            bitmap_j = index_to_bitmap[j]
            bitmap_v, mul = gmt_element(bitmap_i, bitmap_j, signature)
            v = bitmap_to_index[bitmap_v]
            list_ind = i * n + j
            k_list[list_ind] = i
            l_list[list_ind] = v
            m_list[list_ind] = j
            mult_table_vals[list_ind] = mul
    return torch.sparse_coo_tensor(
        indices=coords, values=mult_table_vals, size=(n, n, n)
    )

class CliffordAlgebra(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.register_buffer("metric", torch.as_tensor(metric))
        self.num_bases = len(metric)
        self.bbo = ShortLexBasisBladeOrder(self.num_bases)
        self.dim = len(self.metric)
        self.n_blades = len(self.bbo.grades)
        cayley = (
            construct_gmt(
                self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric
            )
            .to_dense()
            .to(torch.get_default_dtype())
        )
        self.grades = self.bbo.grades.unique()
        self.register_buffer(
            "subspaces",
            torch.tensor(tuple(math.comb(self.dim, g) for g in self.grades)),
        )
        self.n_subspaces = len(self.grades)
        self.grade_to_slice = self._grade_to_slice(self.subspaces)
        self.grade_to_index = [
            torch.tensor(range(*s.indices(s.stop))) for s in self.grade_to_slice
        ]
        self.register_buffer(
            "bbo_grades", self.bbo.grades.to(torch.get_default_dtype())
        )
        self.register_buffer("even_grades", self.bbo_grades % 2 == 0)
        self.register_buffer("odd_grades", ~self.even_grades)
        self.register_buffer("cayley", cayley)

    def geometric_product(self, a, b, blades=None):
        cayley = self.cayley
        if blades is not None:
            blades_l, blades_o, blades_r = blades
            assert isinstance(blades_l, torch.Tensor)
            assert isinstance(blades_o, torch.Tensor)
            assert isinstance(blades_r, torch.Tensor)
            cayley = cayley[blades_l[:, None, None], blades_o[:, None], blades_r]
        return torch.einsum("...i,ijk,...k->...j", a, cayley, b)

    def _grade_to_slice(self, subspaces):
        grade_to_slice = list()
        subspaces = torch.as_tensor(subspaces)
        for grade in self.grades:
            index_start = subspaces[:grade].sum()
            index_end = index_start + math.comb(self.dim, grade)
            grade_to_slice.append(slice(index_start, index_end))
        return grade_to_slice

    @functools.cached_property
    def _alpha_signs(self):
        return torch.pow(-1, self.bbo_grades)

    @functools.cached_property
    def _beta_signs(self):
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades - 1) // 2)

    @functools.cached_property
    def _gamma_signs(self):
        return torch.pow(-1, self.bbo_grades * (self.bbo_grades + 1) // 2)

    def alpha(self, mv, blades=None):
        signs = self._alpha_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def beta(self, mv, blades=None):
        signs = self._beta_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def gamma(self, mv, blades=None):
        signs = self._gamma_signs
        if blades is not None:
            signs = signs[blades]
        return signs * mv.clone()

    def zeta(self, mv):
        return mv[..., :1]

    def embed(self, tensor: torch.Tensor, tensor_index: torch.Tensor) -> torch.Tensor:
        mv = torch.zeros(
            *tensor.shape[:-1], 2 ** self.dim, device=tensor.device, dtype=tensor.dtype
        )
        mv[..., tensor_index] = tensor
        return mv

    def embed_grade(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        mv = torch.zeros(*tensor.shape[:-1], 2 ** self.dim, device=tensor.device)
        s = self.grade_to_slice[grade]
        mv[..., s] = tensor
        return mv

    def get(self, mv: torch.Tensor, blade_index: tuple[int]) -> torch.Tensor:
        blade_index = tuple(blade_index)
        return mv[..., blade_index]

    def get_grade(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        s = self.grade_to_slice[grade]
        return mv[..., s]

    def b(self, x, y, blades=None):
        if blades is not None:
            assert len(blades) == 2
            beta_blades = blades[0]
            blades = (
                blades[0],
                torch.tensor([0]),
                blades[1],
            )
        else:
            blades = torch.tensor(range(self.n_blades))
            blades = (
                blades,
                torch.tensor([0]),
                blades,
            )
            beta_blades = None
        return self.geometric_product(
            self.beta(x, blades=beta_blades),
            y,
            blades=blades,
        )

    def q(self, mv, blades=None):
        if blades is not None:
            blades = (blades, blades)
        return self.b(mv, mv, blades=blades)

    def _smooth_abs_sqrt(self, input, eps=1e-16):
        return (input**2 + eps) ** 0.25

    def norm(self, mv, blades=None):
        return self._smooth_abs_sqrt(self.q(mv, blades=blades))

    def norms(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.norm(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def qs(self, mv, grades=None):
        if grades is None:
            grades = self.grades
        return [
            self.q(self.get_grade(mv, grade), blades=self.grade_to_index[grade])
            for grade in grades
        ]

    def sandwich(self, u, v, w):
        return self.geometric_product(self.geometric_product(u, v), w)

    def output_blades(self, blades_left, blades_right):
        blades = []
        for blade_left in blades_left:
            for blade_right in blades_right:
                bitmap_left = self.bbo.index_to_bitmap[blade_left]
                bitmap_right = self.bbo.index_to_bitmap[blade_right]
                bitmap_out, _ = gmt_element(bitmap_left, bitmap_right, self.metric)
                index_out = self.bbo.bitmap_to_index[bitmap_out]
                blades.append(index_out)
        return torch.tensor(blades)

    def random(self, n=None):
        if n is None:
            n = 1
        return torch.randn(n, self.n_blades)

    def random_vector(self, n=None):
        if n is None:
            n = 1
        vector_indices = self.bbo_grades == 1
        v = torch.zeros(n, self.n_blades, device=self.cayley.device)
        v[:, vector_indices] = torch.randn(
            n, vector_indices.sum(), device=self.cayley.device
        )
        return v

    def parity(self, mv):
        is_odd = torch.all(mv[..., self.even_grades] == 0)
        is_even = torch.all(mv[..., self.odd_grades] == 0)
        if is_odd ^ is_even: # exclusive or (xor)
            return is_odd
        else:
            raise ValueError("This is not a homogeneous element.")

    def eta(self, w):
        return (-1) ** self.parity(w)

    def alpha_w(self, w, mv):
        return self.even_grades * mv + self.eta(w) * self.odd_grades * mv

    def inverse(self, mv, blades=None):
        mv_ = self.beta(mv, blades=blades)
        return mv_ / self.q(mv)

    def rho(self, w, mv):
        """Applies the versor w action to mv."""
        return self.sandwich(w, self.alpha_w(w, mv), self.inverse(w))

    def reduce_geometric_product(self, inputs):
        return functools.reduce(self.geometric_product, inputs)

    def versor(self, order=None, normalized=True):
        if order is None:
            order = self.dim if self.dim % 2 == 0 else self.dim - 1
        vectors = self.random_vector(order)
        versor = self.reduce_geometric_product(vectors[:, None])
        if normalized:
            versor = versor / self.norm(versor)[..., :1]
        return versor

    def rotor(self):
        return self.versor()

    @functools.cached_property
    def geometric_product_paths(self):
        gp_paths = torch.zeros((self.dim + 1, self.dim + 1, self.dim + 1), dtype=bool)
        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                for k in range(self.dim + 1):
                    s_i = self.grade_to_slice[i]
                    s_j = self.grade_to_slice[j]
                    s_k = self.grade_to_slice[k]
                    m = self.cayley[s_i, s_j, s_k]
                    gp_paths[i, j, k] = (m != 0).any()
        return gp_paths

EPS = 1e-6

class NormalizationLayer(nn.Module):
    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features
        self.a = nn.Parameter(torch.zeros(self.in_features, algebra.n_subspaces) + init)

    def forward(self, input):
        assert input.shape[1] == self.in_features
        norms = torch.cat(self.algebra.norms(input), dim=-1)
        s_a = torch.sigmoid(self.a)
        norms = s_a * (norms - 1) + 1 # Interpolates between 1 and the norm.
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        normalized = input / (norms + EPS)
        return normalized

class MVLinear(nn.Module):
    def __init__(self, algebra, in_features, out_features, subspaces=True, bias=True):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces
        if subspaces:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, algebra.n_subspaces)
            )
            self._forward = self._forward_subspaces
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features, 1))
            self.b_dims = (0,)
        else:
            self.register_parameter("bias", None)
            self.b_dims = ()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _forward(self, input):
        return torch.einsum("bm...i, nm->bn...i", input, self.weight)

    def _forward_subspaces(self, input):
        weight = self.weight.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.einsum("bm...i, nmi->bn...i", input, weight)

    def forward(self, input):
        result = self._forward(input)
        if self.bias is not None:
            bias = self.algebra.embed(self.bias, self.b_dims)
            result += unsqueeze_like(bias, result, dim=2)
        return result

class FullyConnectedSteerableGeometricProductLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        include_first_order=True,
        normalization_init=0,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.include_first_order = include_first_order
        if normalization_init is not None:
            self.normalization = NormalizationLayer(
                algebra, in_features, normalization_init
            )
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, in_features, in_features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, in_features, out_features, bias=True)
        self.product_paths = algebra.geometric_product_paths
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, self.product_paths.sum())
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.weight,
            std=1 / math.sqrt(self.in_features * (self.algebra.dim + 1)),
        )

    def _get_weight(self):
        weight = torch.zeros(
            self.out_features,
            self.in_features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, :, self.product_paths] = self.weight
        subspaces = self.algebra.subspaces
        weight_repeated = (
            weight.repeat_interleave(subspaces, dim=-3)
            .repeat_interleave(subspaces, dim=-2)
            .repeat_interleave(subspaces, dim=-1)
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)
        weight = self._get_weight()
        if self.include_first_order:
            return (
                self.linear_left(input)
                + torch.einsum("bni, mnijk, bnk -> bmj", input, weight, input_right)
            ) / math.sqrt(2)
        else:
            return torch.einsum("bni, mnijk, bnk -> bmj", input, weight, input_right)

class SteerableGeometricProductLayer(nn.Module):
    def __init__(
        self, algebra, features, include_first_order=True, normalization_init=0
    ):
        super().__init__()
        self.algebra = algebra
        self.features = features
        self.include_first_order = include_first_order
        if normalization_init is not None:
            self.normalization = NormalizationLayer(
                algebra, features, normalization_init
            )
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, features, features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, features, features, bias=True)
        self.product_paths = algebra.geometric_product_paths
        self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum()))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.algebra.dim + 1))

    def _get_weight(self):
        weight = torch.zeros(
            self.features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, self.product_paths] = self.weight
        subspaces = self.algebra.subspaces
        weight_repeated = (
            weight.repeat_interleave(subspaces, dim=-3)
            .repeat_interleave(subspaces, dim=-2)
            .repeat_interleave(subspaces, dim=-1)
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)
        weight = self._get_weight()
        if self.include_first_order:
            return (
                self.linear_left(input)
                + torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)
            ) / math.sqrt(2)
        else:
            return torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)

class MVLayerNorm(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.a = nn.Parameter(torch.ones(1, channels))

    def forward(self, input):
        norm = self.algebra.norm(input)[..., :1].mean(dim=1, keepdim=True) + EPS
        a = unsqueeze_like(self.a, norm, dim=2)
        return a * input / norm

class MVSiLU(nn.Module):
    def __init__(self, algebra, channels, invariant="mag2", exclude_dual=False):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.exclude_dual = exclude_dual
        self.invariant = invariant
        self.a = nn.Parameter(torch.ones(1, channels, algebra.dim + 1))
        self.b = nn.Parameter(torch.zeros(1, channels, algebra.dim + 1))
        if invariant == "norm":
            self._get_invariants = self._norms_except_scalar
        elif invariant == "mag2":
            self._get_invariants = self._mag2s_except_scalar
        else:
            raise ValueError(f"Invariant {invariant} not recognized.")

    def _norms_except_scalar(self, input):
        return self.algebra.norms(input, grades=self.algebra.grades[1:])

    def _mag2s_except_scalar(self, input):
        return self.algebra.qs(input, grades=self.algebra.grades[1:])

    def forward(self, input):
        norms = self._get_invariants(input)
        norms = torch.cat([input[..., :1], *norms], dim=-1)
        a = unsqueeze_like(self.a, norms, dim=2)
        b = unsqueeze_like(self.b, norms, dim=2)
        norms = a * norms + b
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.sigmoid(norms) * input

def get_invariants(algebra, input):
    norms = algebra.qs(input, grades=algebra.grades[1:])
    return torch.cat([input[..., :1], *norms], dim=-1)

def psi(p):
    """`\psi(p) = Sgn(p) \cdot \log(|p| + 1)`"""
    return torch.sign(p) * torch.log(torch.abs(p) + 1)

def unsorted_segment_sum(data, segment_ids, num_segments):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def _pairwise_deltaR(points_part):
    """ΔR = sqrt(Δη² + Δφ²) with circular φ wrap."""
    eta_diff = points_part[:, None, 0] - points_part[None, :, 0]
    phi_diff = torch.abs(points_part[:, None, 1] - points_part[None, :, 1])
    phi_diff = torch.min(phi_diff, 2 * math.pi - phi_diff)
    return torch.sqrt(eta_diff**2 + phi_diff**2 + 1e-8)


def _pairwise_minkowski(p4_part):
    """
    Lorentz-invariant distance: sqrt(|Δp²| + ε)
    where Δp² = ΔE² - Δpx² - Δpy² - Δpz²  (signature +,-,-,-)
    p4_part: (N, 4) tensor in (E, px, py, pz) order
    """
    diff = p4_part[:, None, :] - p4_part[None, :, :]   # (N, N, 4)
    # Minkowski quadratic form with metric (+,-,-,-)
    mink = (diff[..., 0] ** 2
            - diff[..., 1] ** 2
            - diff[..., 2] ** 2
            - diff[..., 3] ** 2)
    return torch.sqrt(torch.abs(mink) + 1e-8)


def generate_edges_vectorized(mask, points, k, M, device,
                              metric="deltaR", fourmomenta=None):
    """Directed, fully-batched kNN edges. Each real particle connects to its k
    nearest real neighbours: edge (i -> j) means j is a neighbour of i, with i the
    receiver (aggregation index in CGENN) and j the sender. No symmetrization, no
    per-jet Python loop. Assumes M == P (the dense particle count).

    Returns COO edge_index (2, E), node ids offset by b*M into the flat B*M space,
    rows = [receivers ; senders].
    """
    B, P = mask.shape
    mask_bool = mask.bool()

    if k is None:  # fully connected within each jet, no self-loops
        pair = mask_bool[:, :, None] & mask_bool[:, None, :]
        pair = pair & ~torch.eye(P, dtype=torch.bool, device=device)[None]
        b, i, j = pair.nonzero(as_tuple=True)
        return torch.stack([b * M + i, b * M + j])

    # ---- pairwise distance for the whole batch: (B, P, P) ----
    # distances follow the input dtype: a .float() downcast here would silently
    # defeat use_float64 runs (near-tied intervals re-rank under transforms)
    if metric == "minkowski":
        # Δp² = m_i² + m_j² - 2<p_i,p_j>, via a gram matrix (no (B,P,P,4) tensor)
        p4 = fourmomenta
        sig = p4.new_tensor([1.0, -1.0, -1.0, -1.0])              # (E, px, py, pz) metric
        msq = (p4 * p4 * sig).sum(-1)                             # (B, P)
        gram = torch.bmm(p4 * sig, p4.transpose(1, 2))           # (B, P, P)
        dist = torch.sqrt((msq[:, :, None] + msq[:, None, :] - 2 * gram).abs() + 1e-8)
    else:  # deltaR with phi wrap
        eta, phi = points[..., 0], points[..., 1]
        deta = eta[:, :, None] - eta[:, None, :]
        dphi = (phi[:, :, None] - phi[:, None, :]).abs()
        dphi = torch.minimum(dphi, 2 * math.pi - dphi)
        dist = torch.sqrt(deta ** 2 + dphi ** 2 + 1e-8)

    # forbid self-loops and senders that are padded particles
    eye = torch.eye(P, dtype=torch.bool, device=device)[None]
    dist = dist.masked_fill(eye | (~mask_bool)[:, None, :], float("inf"))

    k_actual = min(k, P - 1)
    nbr = dist.topk(k_actual, dim=-1, largest=False).indices      # (B, P, k) senders per receiver

    offset = (torch.arange(B, device=device) * M)[:, None, None]
    recv = (torch.arange(P, device=device)[None, :, None] + offset).expand(B, P, k_actual).reshape(-1)
    send = (nbr + offset).reshape(-1)

    # keep edges with both endpoints real (drops padded senders in sparse jets,
    # which is what a sparse graph wants -- those nodes simply get fewer edges).
    # Also drop self-loops: on jets with n_real <= k, topk fills the remaining
    # slots from the tied +inf pool (self and padded columns), and while padded
    # fills fail the realness filter, a self fill has BOTH endpoints real and
    # would survive as a spurious i->i message (tie-break dependent, hence
    # non-deterministic) -- the same sparse-jet leak fixed in the static-kNN
    # nbr_masks of the Plain/LorentzNet hybrids.
    valid = mask_bool.reshape(-1)
    keep = valid[recv] & valid[send] & (recv != send)
    return torch.stack([recv[keep], send[keep]])

class CGLayer(nn.Module):
    def __init__(
        self,
        algebra,
        in_features_x,
        hidden_features_x,
        out_features_x,
        in_features_h,
        hidden_features_h,
        out_features_h,
        edge_attr_x=3,
        edge_attr_h=0,
        node_attr_x=2,
        node_attr_h=2,
        aggregation="mean",
        use_invariants_to_update=True,
        residual=False,
        normalization_init=None,
        layer_type="fc",
    ):
        super().__init__()
        self.edge_attr_x = edge_attr_x
        self.algebra = algebra
        invariants_h = out_features_x * self.algebra.n_subspaces
        f_in_h = 3 * in_features_h
        self.phi_h = nn.Sequential(
            nn.Linear(
                f_in_h + edge_attr_h + invariants_h,
                hidden_features_h,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
        )
        f_in_x = 3 * in_features_x
        if layer_type == "fc":
            self.phi_x = nn.Sequential(
                FullyConnectedSteerableGeometricProductLayer(
                    self.algebra,
                    edge_attr_x + f_in_x,
                    hidden_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, hidden_features_x),
            )
            self.theta_x = nn.Sequential(
                FullyConnectedSteerableGeometricProductLayer(
                    self.algebra,
                    node_attr_x + in_features_x + hidden_features_x,
                    out_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features_x),
            )
        elif layer_type == "gpmlp":
            self.phi_x = nn.Sequential(
                MVLinear(self.algebra, edge_attr_x + f_in_x, hidden_features_x),
                MVSiLU(self.algebra, hidden_features_x),
                SteerableGeometricProductLayer(
                    self.algebra,
                    hidden_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, hidden_features_x),
            )
            self.theta_x = nn.Sequential(
                MVLinear(
                    self.algebra,
                    node_attr_x + in_features_x + hidden_features_x,
                    out_features_x,
                ),
                MVSiLU(self.algebra, out_features_x),
                SteerableGeometricProductLayer(
                    self.algebra,
                    out_features_x,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features_x),
            )
        else:
            raise ValueError(f"Unknown layer type {layer_type}.")
        self.theta_h = nn.Sequential(
            nn.Linear(
                node_attr_h
                + algebra.n_subspaces * hidden_features_x
                + in_features_h
                + hidden_features_h,
                hidden_features_h,
            ),
            nn.BatchNorm1d(hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_h),
        )
        if aggregation == "mean":
            self.aggregation = unsorted_segment_mean
        elif aggregation == "sum":
            self.aggregation = unsorted_segment_sum
        else:
            raise ValueError(f"Unknown aggregation {aggregation}")
        self.use_invariants_to_update = use_invariants_to_update
        self.residual = residual
        self.layer_type = layer_type
        self.out_features_x = out_features_x
        self.in_features_x = in_features_x
        self.in_features_h = in_features_h
        self.out_features_h = out_features_h
        self.psi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * self.algebra.n_subspaces),
        )
        self.chi_x = nn.Sequential(
            nn.Linear(hidden_features_h, hidden_features_h),
            nn.ReLU(),
            nn.Linear(hidden_features_h, out_features_x * self.algebra.n_subspaces),
        )
        self.aggregation = aggregation

    def reduce(self, input, segment_ids, num_segments):
        if self.aggregation == "mean":
            red = unsorted_segment_mean(input, segment_ids, num_segments=num_segments)
        elif self.aggregation == "sum":
            red = unsorted_segment_sum(input, segment_ids, num_segments=num_segments)
        else:
            raise ValueError(f"Invalid aggregation function {self.aggregation}.")
        return red

    def message_x(self, x_i, x_j, edge_attr_x=None):
        x_diff = x_i - x_j
        input = [x_i, x_j, x_diff]
        if edge_attr_x is not None:
            input.append(edge_attr_x)
        input = torch.cat(input, dim=1)
        return self.phi_x(input)

    def message_h(self, h_i, h_j, invariants_ij, edge_attr_h=None):
        input = [invariants_ij, h_i, h_j, h_i - h_j]
        if edge_attr_h is not None:
            input.append(edge_attr_h)
        input = torch.cat(input, dim=1)
        return self.phi_h(input)

    def update_x(self, x, x_red, node_attr_x):
        if node_attr_x is not None:
            input = torch.cat([x, x_red, node_attr_x], dim=1)
        else:
            input = torch.cat([x, x_red], dim=1)
        return self.theta_x(input)

    def update_h(self, h, h_red, invariants_i, node_attr_h):
        if node_attr_h is not None:
            input = torch.cat([h, h_red, invariants_i, node_attr_h], dim=1)
        else:
            input = torch.cat(
                [
                    h,
                    h_red,
                    invariants_i,
                ],
                dim=1,
            )
        return self.theta_h(input)

    def forward(self, h, x, edges, node_attr_h, node_attr_x, edge_attr_h, edge_attr_x):
        i, j = edges
        m_x = self.message_x(x[i], x[j], edge_attr_x)
        m_invariants = get_invariants(self.algebra, m_x).flatten(1)
        if h is not None:
            m_h = self.message_h(h[i], h[j], m_invariants, edge_attr_h)
        else:
            m_h = None
        if self.use_invariants_to_update:
            weights = self.psi_x(m_h).view(
                len(m_h), self.out_features_x, self.algebra.n_subspaces
            )
            weights = torch.repeat_interleave(weights, self.algebra.subspaces, dim=2)
            m_x = m_x * torch.sigmoid(weights)
        x_red = self.reduce(m_x.flatten(1), i, num_segments=x.size(0)).view(
            len(x), *m_x.shape[1:]
        )
        if m_h is not None:
            h_red = self.reduce(m_h, i, num_segments=h.size(0))
        else:
            h_red = None
        x_u = self.update_x(x, x_red, node_attr_x)
        u_invariants = get_invariants(self.algebra, x).flatten(1)
        if h_red is not None:
            h_u = self.update_h(h, h_red, u_invariants, node_attr_h)
        if self.use_invariants_to_update:
            weights = self.chi_x(h_u).view(
                len(h_u), self.out_features_x, self.algebra.n_subspaces
            )
            weights = torch.repeat_interleave(weights, self.algebra.subspaces, dim=2)
            x_u = x_u * torch.sigmoid(weights)
        if self.residual and self.in_features_h == self.out_features_h:
            h = h_u + h
        else:
            h = h_u
        if self.residual and self.in_features_x == self.out_features_x:
            x = x_u + x
        else:
            x = x_u
        return h, x

class CGENNBackbone(nn.Module):
    def __init__(
        self,
        in_features_h: int = 2,
        hidden_features_h: int = 72,
        in_features_x: int = 1,
        hidden_features_x: int = 8,
        n_layers=4,
        use_invariants_to_update=True,
        normalization_init=None,
        residual=False,
        aggregation="mean",
        layer_type="fc",
    ):
        super().__init__()
        self.in_features_h = in_features_h
        self.hidden_features_h = hidden_features_h
        self.in_features_x = in_features_x
        self.hidden_features_x = hidden_features_x
        self.algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        self.n_layers = n_layers
        self.embedding_h = nn.Linear(in_features_h, hidden_features_h)
        self.embedding_x = MVLinear(
            self.algebra, in_features_x, hidden_features_x, subspaces=False
        )
        self.CGLs = nn.ModuleList(
            [
                CGLayer(
                    self.algebra,
                    hidden_features_x,
                    hidden_features_x,
                    hidden_features_x,
                    hidden_features_h,
                    hidden_features_h,
                    hidden_features_h,
                    # edge_attr_x = 3 copies (diff, i, j) of the raw input mv channels
                    edge_attr_x=1+ 2 * in_features_x,
                    use_invariants_to_update=use_invariants_to_update,
                    normalization_init=normalization_init,
                    residual=residual,
                    aggregation=aggregation,
                    layer_type=layer_type,
                    node_attr_h=in_features_h,
                    node_attr_x=in_features_x,
                )
                for i in range(n_layers)
            ]
        )

    def forward(
        self,
        h,
        x,
        edges,
        node_attr_h=None,
        node_attr_x=None,
        edge_attr_h=None,
        edge_attr_x=None,
    ):
        h = self.embedding_h(h)
        x = self.embedding_x(x)
        for i in range(self.n_layers):
            h, x = self.CGLs[i](
                h,
                x,
                edges,
                node_attr_x=node_attr_x,
                node_attr_h=node_attr_h,
                edge_attr_x=edge_attr_x,
                edge_attr_h=edge_attr_h,
            )
        return h, x

class CGENNLGATrGraphTrans(nn.Module):
    """Hybrid CGENN -> L-GATr model with complementary raw+learned features"""
    def __init__(
        self,
        in_s_channels: int,
        hidden_mv_channels: int,
        hidden_s_channels: int,
        num_classes: int,
        num_blocks: int,
        num_heads: int,
        k: int = None,
        cgenn_layers: int = 2,
        cgenn_hidden_h: int = 72,
        cgenn_hidden_x: int = 8,
        cgenn_aggregation: str = "mean",
        cgenn_residual: bool = False,  # official CGENN top-tagging default (tag_cgenn row)
        cgenn_layer_type: str = "fc",
        cgenn_normalization_init=None,  # official CGENN: no NormalizationLayer (tag_cgenn row)
        concat_original: bool = True,
        use_explicit_edge_features: bool = True,
        beam_spurion: str = "xyplane",
        add_time_spurion: bool = True,
        beam_mirror: bool = True,
        knn_metric: str = "deltaR",
        activation: str = "gelu",
        multi_query: bool = False,
        increase_hidden_channels_attention: int = 2,
        increase_hidden_channels_mlp: int = 2,
        num_hidden_layers_mlp: int = 1,
        head_scale: bool = False,
        dropout_prob: float = None,
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, -1.0, -1.0, -1.0))
        self.hidden_mv_channels = hidden_mv_channels
        self.in_s_channels = in_s_channels
        self.concat_original = concat_original
        self.use_explicit_edge_features = use_explicit_edge_features
        if not use_explicit_edge_features:
            # The CGENN stage's CGLayers are built expecting the relative-momentum edge
            # multivectors and the re-injected raw node attributes (edge_attr_x / node_attr_x /
            # node_attr_h with nonzero dims); the forward would pass None for them, mismatching
            # the layer dimensions -> a shape RuntimeError on the first layer. Fail loudly here
            # instead. (The GraphGPS sibling, cgennlgatrgraphgps.py, DOES support the toggle --
            # it zeroes those dims at construction when the flag is off.)
            raise NotImplementedError(
                "CGENNLGATrGraphTrans supports only use_explicit_edge_features=True; the CGENN "
                "stage is constructed with the static edge/node attributes always enabled. Use "
                "CGENNLGATrGraphGPS if you need the no-edge-feature ablation."
            )
        self.spurion_kwargs = {
            "beam_spurion": beam_spurion,
            "add_time_spurion": add_time_spurion,
            "beam_mirror": beam_mirror,
        }
        num_spurions = get_num_spurions(
            beam_spurion, add_time_spurion, beam_mirror=beam_mirror
        )
        self.num_spurions = num_spurions
        self.k = k
        self.knn_metric = knn_metric
        if knn_metric not in ("deltaR", "minkowski"):
            raise ValueError(f"knn_metric must be 'deltaR' or 'minkowski', got '{knn_metric}'")

        # Spurions are injected as extra mv input channels (1 particle channel +
        # num_spurions spurion channels), so no spurion tokens enter the graph.
        in_mv_channels_cgenn = 1 + num_spurions

        self.cgenn = CGENNBackbone(
            in_features_h=in_s_channels,
            hidden_features_h=cgenn_hidden_h,
            in_features_x=in_mv_channels_cgenn,
            hidden_features_x=cgenn_hidden_x,
            n_layers=cgenn_layers,
            use_invariants_to_update=True,
            normalization_init=cgenn_normalization_init,
            residual=cgenn_residual,
            aggregation=cgenn_aggregation,
            layer_type=cgenn_layer_type,
        )

        # concat_original "skips" the raw particle kinematic channel (ch 0) only.
        # Spurion channels are intentionally excluded: they are global constants
        # (zero variance across the batch) whose information is already folded
        # into every CGENN output channel via embedding_x. Concatenating them
        # here would add identical constant MVs to every token — no new signal.
        mv_bridge_in = cgenn_hidden_x + 1 if concat_original else cgenn_hidden_x
        self.mv_bridge = MVLinear(self.algebra, mv_bridge_in, hidden_mv_channels, subspaces=True)

        s_bridge_in = cgenn_hidden_h + (in_s_channels if concat_original else 0)  # <-- wider
        self.s_bridge = nn.Linear(s_bridge_in, hidden_s_channels)

        self.cls_mv_scalar = nn.Parameter(torch.zeros(1, 1, hidden_mv_channels))
        torch.nn.init.normal_(self.cls_mv_scalar, std=0.02)
        self.cls_s = nn.Parameter(torch.zeros(1, 1, hidden_s_channels))
        torch.nn.init.normal_(self.cls_s, std=0.02)

        attention = dict(
            multi_query=multi_query,
            num_heads=num_heads,
            increase_hidden_channels=increase_hidden_channels_attention,
            head_scale=head_scale,
        )
        mlp = dict(
            activation=activation,
            increase_hidden_channels=increase_hidden_channels_mlp,
            num_hidden_layers=num_hidden_layers_mlp,
        )
        self.net = LGATr(
            num_blocks=num_blocks,
            in_mv_channels=hidden_mv_channels,
            out_mv_channels=num_classes,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            attention=attention,
            mlp=mlp,
            dropout_prob=dropout_prob,
            checkpoint_blocks=checkpoint_blocks,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        # both CLS tokens are learnable invariants; excluding them from weight
        # decay is the ParT/ViT convention and does not affect equivariance
        return {
            "cls_mv_scalar",
            "cls_s",
        }

    def forward(self, x, v, mask, points):
   # points-first inputs from the wrapper:
        #   x: (B, P, C)   v: (B, P, 4) [E, px, py, pz]   mask: (B, P)   points: (B, P, 2)

        # Stage 1: Multivector embedding
        fourmomenta_ga = v[:, :, None, :]       # (B, P, 1, 4)  # Stage 1; was v[:, :, None, [3, 0, 1, 2]]
        mv = embed_vector(fourmomenta_ga)               # (B, P, 1, 16)
        s = x                                           # (B, P, C)

        # Stage 2: Inject spurions as extra mv channels (not tokens)
        # Each spurion is broadcast to every particle slot as an additional
        # input channel, giving CGENN access to the symmetry-breaking axes
        # without adding any nodes to the graph.
        device = s.device
        if self.num_spurions > 0:
            spurions = get_spurions(**self.spurion_kwargs).to(device=device, dtype=s.dtype)
            # spurions: (num_spurions, 16)  →  (B, P, num_spurions, 16)
            spurion_channels = spurions[None, None, :, :].expand(
                mv.shape[0], mv.shape[1], -1, -1
            )
            mv = torch.cat([mv, spurion_channels], dim=2)   # (B, P, 1+num_spurions, 16)
        # s and mask are unmodified — no spurion tokens

        B, P, _ = s.shape
        M = P

        # Stage 3: Build graph edges (native dtype: see generate_edges_vectorized)
        fourmomenta_flat = None
        if self.knn_metric == "minkowski" and self.k is not None:
            fourmomenta_flat = v

        edges = generate_edges_vectorized(
            mask,
            points,
            self.k,
            M,
            device,
            metric=self.knn_metric,
            fourmomenta=fourmomenta_flat,
        )

        # Stage 4: Flatten for CGENN over the dense B*P layout (padded slots
        # included), matching official CGENN: its theta_h BatchNorm also runs over
        # the padded nodes. Edges connect real nodes only, so padding never
        # propagates through message passing -- it only enters the scalar BN stats,
        # exactly as upstream.
        total_nodes = B * M
        h_flat = s.reshape(total_nodes, -1)
        x_flat_raw = mv.reshape(total_nodes, -1, 16)  # (B*P, 1+num_spurions, 16)

        if self.use_explicit_edge_features:
            i, j = edges
            particle_diff = x_flat_raw[i, :1] - x_flat_raw[j, :1]
            edge_attr_x = torch.cat([particle_diff, x_flat_raw[i], x_flat_raw[j]], dim=1)

            node_attr_x = x_flat_raw
            node_attr_h = h_flat
        else:
            edge_attr_x = None
            node_attr_x = None
            node_attr_h = None

        # Stage 5: CGENN layers
        h_flat, x_flat = self.cgenn(
            h_flat,
            x_flat_raw,
            edges,
            node_attr_h=node_attr_h,
            node_attr_x=node_attr_x,
            edge_attr_h=None,
            edge_attr_x=edge_attr_x,
        )

        # Reshape back
        h = h_flat.view(B, M, -1)
        x = x_flat.view(B, M, -1, 16)

        # Stage 6: Linear bridge
        if self.concat_original:
            # "Skip-connect" raw particle kinematics (channel 0) only.
            # Spurion channels (1..num_spurions) are excluded: they are fixed
            # constants with no per-particle variance. Their contribution is
            # already encoded in every CGENN output channel via embedding_x,
            # so repeating them here would be redundant and waste bridge capacity.
            particle_mv = mv[:, :, :1, :]                  # (B, P, 1, 16)
            x = torch.cat([particle_mv, x], dim=2)         # (B, P, 1+hidden_x, 16)
            h = torch.cat([s, h], dim=2) 

        x_bridge = x.reshape(B * M, -1, 16)
        h_bridge = h.reshape(B * M, -1)

        mv_out = self.mv_bridge(x_bridge).view(B, M, -1, 16)
        s_out = self.s_bridge(h_bridge).view(B, M, -1)

        # Stage 7: Add learnable CLS token (does this break equivariance?)
        cls_mv = torch.zeros(B, 1, self.hidden_mv_channels, 16, device=device, dtype=s_out.dtype)
        cls_mv[..., 0] = self.cls_mv_scalar.expand(B, 1, -1)
        cls_s = self.cls_s.expand(B, -1, -1)
        cls_mask = torch.ones(B, 1, device=device, dtype=torch.bool)
#tests say no
        mv_out = torch.cat([cls_mv, mv_out], dim=1)
        s_out = torch.cat([cls_s, s_out], dim=1)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Stage 8: L-GATr Transformer
        attn_mask = mask[:, None, None, :]
        out_mv, _ = self.net(mv_out, s_out, attn_mask=attn_mask)

        # Stage 9: Classification from CLS token
        cls_out = out_mv[:, 0]
        output = extract_scalar(cls_out)[..., 0]

        return output
     
