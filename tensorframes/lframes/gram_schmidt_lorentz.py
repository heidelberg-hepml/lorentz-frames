import warnings
from typing import Union

import torch
from torch import Tensor


def leinsum(einstr:str, a: torch.Tensor, b: torch.Tensor, dim: int = -1):
    index = [slice(None)] * b.dim()

    index[dim] = 0

    b_copy = b.detach().clone()
    b_copy[index] *= -1
    return torch.einsum(einstr, a, b_copy)

def gram_schmidt_lorentz(
    vectors,
    eps: float = 1e-6,
    normalized: bool = True,
    exceptional_choice: str = "random",
) -> Tensor:
    """Applies the Gram-Schmidt process to a set of input vectors to orthogonalize them.

    Args:
        vectors (Tensor): The input vectors. shape (4, 4, N) (vectors, dims, size)
        eps (float, optional): A small value used for numerical stability. Defaults to 1e-6.
        normalized (bool, optional): Whether to normalize the output vectors. Defaults to True.
        exceptional_choice (str, optional): The method to handle exceptional cases where the input vectors have zero length.
            Can be either "random" to use a random vector instead, or "zero" to set the vectors to zero.
            Defaults to "random".

    Returns:
        Tensor: A tensor containing the orthogonalized vectors the tensor has shape (4, 4, N).

    Raises:
        ValueError: If the exceptional_choice parameter is not recognized.
        AssertionError: If z_axis has zero length.
    """
    
    assert normalized == True
    
    
    assert vectors.shape[:-1] == (4,4) or vectors.shape[:-1] == (3, 4), f"The shape of the given vectors ({vectors.shape}) needs to be in the format (vectors, dim, size), where vectors can be 4 or 3, dim has to be 4 and size is arbitrary"
    vec = vectors.clone()
    errorCounter = 0
    for index in range(vec.shape[0]):
        error = True
        while error==True:
            sign = leinsum("vds,vds->vs", vec[:index], vec[:index], dim=-2).sign().unsqueeze(-2)
            weights = leinsum("vds,ds->vs",sign*vec[:index], vec[index],dim=-2)
            normBefore =  leinsum("ds,ds->s", vec[index], vec[index], dim=-2).abs().sqrt()
            vec[index] -= torch.sum(torch.einsum("vs,vds->vds", weights, vec[:index]), axis=0)
            norm = leinsum("ds,ds->s", vec[index], vec[index], dim=-2).abs().sqrt()
            zeroNorm = norm < 2.e-1*normBefore
            if zeroNorm.sum().item()!=0: # linearly alligned elements / zero norm
                vec[index, :, zeroNorm] = torch.rand(vec[index, :, zeroNorm].shape)
                errorCounter += 1
            else:
                vec[index] /= norm.unsqueeze(-2)
                error=False
    if vectors.shape[:-1] == (3,4):
        x, y, z = vec
        alpha1 = y[1]/x[1]
        alpha2 = z[1]/x[1]
        beta = (z[2]-alpha2*x[2]) / (y[2]-alpha1*x[2])
        
        vec = torch.cat((vec, torch.zeros(1,vectors.shape[1], vectors.shape[-1])), dim=0)
        
        vec[-1,3] = ((z[0]-alpha2*x[0])-beta*(y[0]-alpha1*x[0])) / ( (z[3]-alpha2*x[3]) - beta*(y[3]-alpha1*x[3]) )
        vec[-1,2] = ( (y[0]-alpha1*x[0]) - (y[3]-alpha1*x[3]) * vec[-1,3]) / (y[2]-alpha1*x[2])
        vec[-1,1] = ( x[0] - x[2]*vec[-1,2] - x[3]*vec[-1,3] ) / x[1]
        vec[-1,0] = torch.tensor(1).repeat( vec.shape[-1] ) 
        norm = leinsum("ds,ds->s", vec[-1], vec[-1], dim=-2).abs().sqrt()
        vec[-1] /= norm.unsqueeze(-2)
    return vec