'''
ResNet based models Mixins
'''
from tensorflow.keras import layers
from tensorflow_addons.layers.adaptive_pooling import AdaptiveAveragePooling2D
from uncertainty_baselines.models.resnet50_deterministic import resnet50_deterministic
#from uncertainty_baselines.models.resnet50_sngp import resnet50_sngp
from shift.tf.model.resnet50_sn import resnet50_sn
from shift.tf.model.resnet50_hetsn import resnet50_hetsn

from shift.tf.model.abstract_models import (
    ModelFactory,
    HetSNGPMixin,
    SNGPMixin,
)

class ResNet50ModelFactory(ModelFactory):
    """
    ModelFactory for a RestNet50 backbone.
    """

    def get_resnet50(self, shape):
        raise NotImplementedError

    def get_backbone(self, shape):
        backbone_mdl = self.get_resnet50(shape)

        backbone_pool = AdaptiveAveragePooling2D((1, 1))(backbone_mdl.output)
        backbone_features = layers.Flatten()(backbone_pool)
        agent_state_vector = layers.Input((3))
        logits = layers.Concatenate()([backbone_features, agent_state_vector])

        return {'image': backbone_mdl.input, 'state': agent_state_vector}, logits


class SNResNet50():
    """
    Deterministic spectral normalized ResNet50 Backbone
    """
    def get_backbone(self, shape):
        backbone_mdl = self.get_resnet50(shape)
        backbone_pool = AdaptiveAveragePooling2D((1, 1))(backbone_mdl.output)
        backbone_features = layers.Flatten()(backbone_pool)
        agent_state_vector = layers.Input(shape=(3), batch_size=self.hyperparameters["batch_size"]) # static batch size necessary for spectral normalization
        logits = layers.Concatenate()([backbone_features, agent_state_vector])

        return {'image': backbone_mdl.input, 'state': agent_state_vector}, logits

    def get_resnet50(self, shape):
        backbone_mdl = resnet50_hetsn(
            input_shape=shape,
            batch_size=self.hyperparameters["batch_size"],
            num_classes=None,
            num_factors=None,
            temperature=None,
            use_mc_dropout=False,
            dropout_rate=0.0,
            filterwise_dropout=True,
            use_gp_layer=False,
            gp_hidden_dim=None,
            gp_scale=None,
            gp_bias=0,
            gp_input_normalization=False,
            gp_random_feature_type='orf',
            gp_cov_discount_factor=-1,
            gp_cov_ridge_penalty=1,
            gp_output_imagenet_initializer=True,
            use_spec_norm=self.hyperparameters["use_spec_norm"],
            spec_norm_iteration=self.hyperparameters["spec_norm_iteration"],
            spec_norm_bound=self.hyperparameters["spec_norm_bound"],
            omit_last_layer=True,
        )
        return backbone_mdl


class ResNet50SNGPMixin(SNResNet50, SNGPMixin):
    """
    Deterministic ResNet50 model with spectral normalization and RFF/Laplace approximated GP head
    """
class ResNet50HETSNGPMixin(SNResNet50, HetSNGPMixin):
    """
    Deterministic ResNet50 model with heteroscedastic spectral normalization and RFF/Laplace approximated GP head
    """