# ##############################################################################
# # IMPORTS

# # THIRD PARTY

# # LOCAL

# ##############################################################################
# # TESTS
# ##############################################################################


# class TestSelfOrganizingMap1D:
#     """Test `trackstream.som.SelfOrganizingMap1D`."""

#     def setup_class(self):
#         """Setup class for testing."""

#     @pytest.fixture
#     def som(self):
#         return self.som_cls(
#             self.nlattice,
#             self.nfeature,

#     # Method Tests

#     def test_init(self):
#         # minimum mandatory arguments

#         # different arguments

#     # ---------------------------------------------------------------
#     # properties

#     def test_nlattice(self, som):
#         assert som.nlattice is som._nlattice

#     def test_nfeature(self, som):
#         assert som.nfeature is som._nfeature

#     def test_sigma(self, som):
#         assert som.sigma is som._sigma

#     def test_learning_rate(self, som):
#         assert som.learning_rate is som._learning_rate

#     def test_decay_function(self, som):
#         assert som.decay_function is som._decay_function
#         assert som.decay_function is asymptotic_decay

#     # ---------------------------------------------------------------
#     # methods


#     def test_activation_distance(self, som):

#     def test_distance_from_weights(self, som):
#         """Test ``SelfOrganizingMap1D._distance_from_weights``."""
#         # test when everything aligns


#         # and errors when not
#         with pytest.raises(ValueError, match="shapes"):

#     @pytest.mark.skip("TODO!")
#     def test_quantization(self, som):
#         """Test ``SelfOrganizingMap1D.quantization``."""
#         assert False

#     @pytest.mark.skip("TODO!")
#     def test_quantization_error(self, som):
#         """Test ``SelfOrganizingMap1D.quantization_error``."""
#         assert False

#     @pytest.mark.skip("TODO!")
#     def test_fit(self, som):
#         assert False

#     @pytest.mark.skip("TODO!")
#     def test_neighborhood(self, som):
#         assert False

#     @pytest.mark.skip("TODO!")
#     def test_update(self, som):
#         assert False

#     @pytest.mark.skip("TODO!")
#     def test_winner(self, som):
#         assert False


# # /class

# #####################################################################


# @pytest.mark.parametrize("learning_rate", [0, 1, 2])
# @pytest.mark.parametrize("iteration", [0, 1, 2])
# @pytest.mark.parametrize("max_iter", [10, 10, 10])
# def test_asymptotic_decay(learning_rate, iteration, max_iter):


# @pytest.mark.skip("TODO!")
# def test_reorder_visits():
#     assert False


# #     assert False
# #
# #
# # # /def


# @pytest.mark.skip("TODO!")
# def test_order_data():
#     assert False
