from abc import ABCMeta, abstractmethod




class DSE(metaclass=ABCMeta):
    """Distributed Structure Embedder (DSE): this is the abstract class for all the structure embedders.

    Given a structure :math:`s`, the main aim of a DSE is to  produce its embedding :math:`\\vec{s} \\in R^d`
    in a reduced space :math:`R^d` which represents :math:`s` by its substrucures.
    This is obtained by obtaining :math:`\\vec{s} \\in R^d` as a weighted sum of
    all the substructures :math:`\\sigma` foreseen in :math:`S(s)`
    according to the specific structure embedder:

    .. math::     \\vec{s} = \\sum_{\\sigma \\in S(s)} \\omega_\\sigma W\\vec{e}_{\\sigma} = \\sum_{\\sigma \\in S(s)} \\omega_\\sigma \\vec{\\sigma} = W\\sum_{\\sigma \\in S(s)} \\omega_\\sigma \\vec{e}_{\\sigma} = W\\vec{e_{s}}
                   :label: main

    where :math:`W` is the embedding matrix, :math:`\\vec{e}_{\\sigma}` is the one-hot vector of
    the substructure :math:`\\sigma` in the space :math:`R^n`, :math:`\\omega_{\\sigma}` is the scalar weight
    of the substructure :math:`\\sigma`, :math:`\\vec{\sigma}` is the vector representing
    the structure :math:`\\sigma` in the smaller space :math:`\\vec{\sigma} = W\\vec{e}_{\\sigma}\\in R^d`,
    and  :math:`\\vec{e_{s}} = \\sum_{\\sigma \\in S(s)} \\omega_{\\sigma}\\vec{e_{\\sigma}}` is the vector representing
    the initial structure :math:`s` in the space :math:`R^n`.
    """

    @abstractmethod
    def ds(self, structure):
        """
        given a structure :math:`s`, produces the embedding :math:`\\vec{s} \\in R^d` (see Equation :eq:`main`).
        This is generally obtained recursively on the components of the structure
        plus a composing operation.

        :param structure: this is the structure :math:`s` that has to be decomposed in substructures.
        :return: :math:`\\vec{s}` which is an Array or a Tensor according to the implementation.
        """
        return None

    @abstractmethod
    def dsf_with_weight(self, structure):
        """given a structure :math:`\\sigma`, it produces the vector :math:`\\vec{\\sigma}` representing that single structure
        in the embedding space :math:`R^d` and its weight :math:`\\omega_{\\sigma}` (see Equation :eq:`main`).

        :param structure: this is the structure that has to be transformed in a vector representing only that structure.
        :return: (:math:`\\vec{\\sigma}`, :math:`\\omega_{\\sigma}`)
        """
        return None

    @abstractmethod
    def dsf(self, structure):
        """given a structure :math:`\\sigma`, it produces the vector :math:`\\vec{\sigma} = W\\vec{e}_{\\sigma}\\in R^d`
        representing that single structure in the embedding space :math:`R^d` (see Equation :eq:`main`).
        This is generally obtained recursively on the components of the structure
        plus a composing operation .

        :param structure: this is the structure :math:`\\sigma` that has to be transformed in a vector representing only that structure.
        :return: :math:`\\vec{\sigma}`
        """
        return None

    @abstractmethod
    def substructures(self, structure):
        """it produces all the substructures of :math:`s` foreseen according to the specific structure embedder, that is,
        :math:`S(s)` (see Equation :eq:`main`)

        :param structure: this is the structure where substructure are extracted from.
        :return: :math:`S(s)` as a list
        """
        return None
