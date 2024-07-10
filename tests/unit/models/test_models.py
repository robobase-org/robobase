import numpy as np
import pytest
import torch

from robobase.models.diffusion_models import ConditionalUnet1D, TransformerForDiffusion
from robobase.models.encoder import EncoderMultiViewVisionTransformer
from robobase.models import (
    EncoderCNNMultiViewDownsampleWithStrides,
    EncoderMVPMultiView,
    FusionMultiCamFeatureAttention,
    FusionMultiCamFeature,
    ResNetEncoder,
)
from robobase.models.fully_connected import (
    MLPWithBottleneckFeatures,
    MLPWithBottleneckFeaturesAndWithoutHead,
    MLPWithBottleneckFeaturesAndSequenceOutput,
)
from robobase.models.decoder import DecoderCNNMultiView
from robobase.method.act import ImageEncoderACT
from robobase.models.multi_view_transformer import MultiViewTransformerEncoderDecoderACT

BATCH_SIZE = 16
SINGLE_CAM = (1, 3, 224, 224)
MULTI_CAM = (2, 3, 224, 224)
MULTI_CAM_AND_CH = (2, 6, 224, 224)

DECODER_SINGLE_CAM = (1, 3, 256, 256)
DECODER_MULTI_CAM = (2, 3, 256, 256)
DECODER_MULTI_CAM_AND_CH = (2, 6, 256, 256)


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM, MULTI_CAM_AND_CH],
)
def test_encoder_cnn_multi_view_with_downsample(input_shape):
    net = EncoderCNNMultiViewDownsampleWithStrides(input_shape, 2, 0, channels=32)
    out = net(torch.rand((BATCH_SIZE,) + input_shape))
    assert out.shape == (BATCH_SIZE, input_shape[0], 32 * 55 * 55)


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM, MULTI_CAM_AND_CH],
)
def test_encoder_cnn_multi_view_with_downsample_and_post(input_shape):
    net = EncoderCNNMultiViewDownsampleWithStrides(input_shape, 2, 2, channels=32)
    out = net(torch.rand((BATCH_SIZE,) + input_shape))
    assert out.shape == (BATCH_SIZE, input_shape[0], 32 * 51 * 51)


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM, MULTI_CAM_AND_CH],
)
def test_encoder_cnn_multi_view_with_different_kernel_sizes(input_shape):
    net = EncoderCNNMultiViewDownsampleWithStrides(
        input_shape, 2, 2, channels=32, kernel_size=4
    )
    out = net(torch.rand((BATCH_SIZE,) + input_shape))
    assert out.shape == (BATCH_SIZE, input_shape[0], 32 * 48 * 48)

    net = EncoderCNNMultiViewDownsampleWithStrides(
        input_shape, 2, 2, channels=32, kernel_size=2
    )
    out = net(torch.rand((BATCH_SIZE,) + input_shape))
    assert out.shape == (BATCH_SIZE, input_shape[0], 32 * 54 * 54)


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM],
)
def test_encoder_mvp_multi_view(input_shape):
    net = EncoderMVPMultiView(input_shape, name="vits-mae-hoi")
    out = net(torch.rand((BATCH_SIZE,) + input_shape))
    assert out.shape == (BATCH_SIZE, input_shape[0], 384)


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM],
)
def test_encoder_resnet(input_shape):
    net = ResNetEncoder(input_shape, "resnet18")
    out = net(torch.rand((BATCH_SIZE,) + input_shape))
    assert out.shape == (BATCH_SIZE, input_shape[0], 512)


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM],
)
def test_encoder_multi_view_vision_transformer(input_shape):
    patch_size = 16
    embed_dim = 256
    num_patches = (input_shape[-1] // patch_size) ** 2
    num_views = input_shape[0]
    net = EncoderMultiViewVisionTransformer(input_shape, patch_size)
    out = net(torch.rand((BATCH_SIZE,) + input_shape))
    assert out.shape == (BATCH_SIZE, num_views, num_patches * embed_dim)


def test_fusion_multi_cam_feature_flatten():
    cams = 3
    feature_size = 64
    mode = "flatten"
    net = FusionMultiCamFeature((cams, feature_size), mode)
    out = net(torch.rand((BATCH_SIZE, cams, feature_size)))
    assert out.shape == (BATCH_SIZE, cams * feature_size)


def test_fusion_multi_cam_feature_sum():
    cams = 3
    feature_size = 64
    mode = "sum"
    net = FusionMultiCamFeature((cams, feature_size), mode)
    out = net(torch.rand((BATCH_SIZE, cams, feature_size)))
    assert out.shape == (BATCH_SIZE, feature_size)


def test_fusion_multi_cam_feature_average():
    cams = 3
    feature_size = 64
    mode = "average"
    net = FusionMultiCamFeature((cams, feature_size), mode)
    out = net(torch.rand((BATCH_SIZE, cams, feature_size)))
    assert out.shape == (BATCH_SIZE, feature_size)


def test_fusion_multi_cam_feature_attention():
    cams = 3
    token_size = 64
    hidden_size = 128
    net = FusionMultiCamFeatureAttention((cams, token_size), hidden_size)
    out = net(torch.rand((BATCH_SIZE, cams, token_size)))
    assert out.shape == (BATCH_SIZE, hidden_size)


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM],
)
def test_image_encoder_act(input_shape):
    net = ImageEncoderACT(input_shape)
    input_data = torch.rand((BATCH_SIZE,) + input_shape)
    out = net(input_data)[0]

    expected_shape = (BATCH_SIZE,) + net.output_shape[0]
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM],
)
def test_image_encoder_act_lang_tokens(input_shape):
    net = ImageEncoderACT(input_shape, use_lang_cond=True)
    lang_toks = torch.rand((BATCH_SIZE, 512))
    input_data = torch.rand((BATCH_SIZE,) + input_shape)
    img_feat, pos_emb, task_emb = net(input_data, task_emb=lang_toks)

    expected_shape = (BATCH_SIZE,) + net.output_shape[0]
    assert img_feat.shape == expected_shape
    assert task_emb is not None


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM],
)
def test_multi_view_transformer_encoder_decoder_act(input_shape):
    img_enc = ImageEncoderACT(input_shape=input_shape)
    input_data = torch.rand((BATCH_SIZE,) + input_shape)
    img_feat, pos_emb, _ = img_enc(input_data)

    for action_sequence in range(1, 5):
        net = MultiViewTransformerEncoderDecoderACT(
            input_shape=input_shape, num_queries=action_sequence
        )

        qpos = torch.rand((BATCH_SIZE, net.state_dim))
        actions = torch.rand((BATCH_SIZE, net.num_queries, net.state_dim))
        is_pad = torch.randint(0, 2, (BATCH_SIZE, net.num_queries)).bool()

        out = net((img_feat, pos_emb), qpos, actions, is_pad)

        expected_shape = (
            (BATCH_SIZE, net.num_queries, net.state_dim),
            (BATCH_SIZE, net.num_queries, 1),
            (BATCH_SIZE, net.latent_dim),
        )

        assert out[0].shape == expected_shape[0]
        assert out[1].shape == expected_shape[1]
        assert out[2][0].shape == expected_shape[2]
        assert out[2][1].shape == expected_shape[2]


@pytest.mark.parametrize(
    "input_shape",
    [SINGLE_CAM, MULTI_CAM],
)
def test_multi_view_transformer_encoder_decoder_act_lang_tokens(input_shape):
    img_enc = ImageEncoderACT(input_shape=input_shape, use_lang_cond=True)
    input_data = torch.rand((BATCH_SIZE,) + input_shape)
    lang_toks = torch.rand((BATCH_SIZE, 512))
    img_feat, pos_emb, task_emb = img_enc(input_data, task_emb=lang_toks)

    for action_sequence in range(1, 5):
        net = MultiViewTransformerEncoderDecoderACT(
            input_shape=input_shape, num_queries=action_sequence, use_lang_cond=True
        )

        qpos = torch.rand((BATCH_SIZE, net.state_dim))
        actions = torch.rand((BATCH_SIZE, net.num_queries, net.state_dim))
        is_pad = torch.randint(0, 2, (BATCH_SIZE, net.num_queries)).bool()

        out = net((img_feat, pos_emb), qpos, actions, is_pad, task_emb=task_emb)

        expected_shape = (
            (BATCH_SIZE, net.num_queries, net.state_dim),
            (BATCH_SIZE, net.num_queries, 1),
            (BATCH_SIZE, net.latent_dim),
        )

        assert out[0].shape == expected_shape[0]
        assert out[1].shape == expected_shape[1]
        assert out[2][0].shape == expected_shape[2]
        assert out[2][1].shape == expected_shape[2]


def test_fully_connected_mlp_with_bottleneck_features():
    in0 = torch.rand((BATCH_SIZE, 2))
    in1 = torch.rand((BATCH_SIZE, 4))
    input_shapes = {"in0": in0.shape[-1:], "in1": in1.shape[-1:]}
    output_shape = 1
    net = MLPWithBottleneckFeatures(
        input_shapes=input_shapes,
        output_shape=output_shape,
        num_envs=1,  # Not used when input_shape n-dim == 1
        rnn_hidden_size=1,  # Not used when input_shape n-dim == 1
        num_rnn_layers=1,  # Not used when input_shape n-dim == 1
        keys_to_bottleneck=["in0"],
        bottleneck_size=5,
        norm_after_bottleneck=True,
        tanh_after_bottleneck=True,
        mlp_nodes=[16, 16],
    )
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (BATCH_SIZE, output_shape)


def test_fully_connected_mlp_with_bottleneck_features_and_without_head():
    in0 = torch.rand((BATCH_SIZE, 2))
    in1 = torch.rand((BATCH_SIZE, 4))
    mlp_nodes = [16, 16]
    input_shapes = {"in0": in0.shape[-1:], "in1": in1.shape[-1:]}
    net = MLPWithBottleneckFeaturesAndWithoutHead(
        input_shapes=input_shapes,
        mlp_nodes=[16, 16],
        num_envs=1,  # Not used when input_shape n-dim == 1
        rnn_hidden_size=1,  # Not used when input_shape n-dim == 1
        num_rnn_layers=1,  # Not used when input_shape n-dim == 1
        output_shape=1,  # Not used when we use WithoutHead class
        keys_to_bottleneck=["in0"],
        bottleneck_size=5,
        norm_after_bottleneck=True,
        tanh_after_bottleneck=True,
    )
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (BATCH_SIZE, mlp_nodes[-1])


def test_fully_connected_mlp_with_bottleneck_features_and_rnn_ins():
    TIME_DIM = 2
    NUM_ENVS = 2
    input_shapes = {"in0": (TIME_DIM, 2), "in1": (TIME_DIM, 4)}
    output_shape = 1
    net = MLPWithBottleneckFeatures(
        input_shapes=input_shapes,
        output_shape=output_shape,
        num_envs=NUM_ENVS + 1,  # for eval
        rnn_hidden_size=32,
        num_rnn_layers=3,
        keys_to_bottleneck=["in0"],
        bottleneck_size=5,
        norm_after_bottleneck=True,
        tanh_after_bottleneck=True,
        mlp_nodes=[16, 16],
    )
    # When in eval mode, BATCH_SIZE must be num_envs
    net.eval()
    in0 = torch.rand((NUM_ENVS, TIME_DIM, 2))
    in1 = torch.rand((NUM_ENVS, TIME_DIM, 4))
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (NUM_ENVS, output_shape)
    # When in train mode, BATCH_SIZE can be anything
    net.train()
    in0 = torch.rand((BATCH_SIZE, TIME_DIM, 2))
    in1 = torch.rand((BATCH_SIZE, TIME_DIM, 4))
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (BATCH_SIZE, output_shape)
    # Reset agent 0 should only clear the state of agent 0
    net.reset(0)
    assert np.all(net.hidden_state[:, 0].numpy() == 0)
    assert not np.all(net.hidden_state[:, 1].numpy() == 0)


def test_fully_connected_mlp_with_bottleneck_features_and_partial_rnn_ins():
    TIME_DIM = 2
    NUM_ENVS = 2
    input_shapes = {"in0": (TIME_DIM, 2), "in1": (4,)}
    output_shape = 1
    net = MLPWithBottleneckFeatures(
        input_shapes=input_shapes,
        output_shape=output_shape,
        num_envs=NUM_ENVS + 1,  # for eval
        rnn_hidden_size=32,
        num_rnn_layers=3,
        keys_to_bottleneck=["in0"],
        bottleneck_size=5,
        norm_after_bottleneck=True,
        tanh_after_bottleneck=True,
        mlp_nodes=[16, 16],
    )
    # When in eval mode, BATCH_SIZE must be num_envs
    net.eval()
    in0 = torch.rand((NUM_ENVS, TIME_DIM, 2))
    in1 = torch.rand((NUM_ENVS, 4))
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (NUM_ENVS, output_shape)
    # When in train mode, BATCH_SIZE can be anything
    net.train()
    in0 = torch.rand((BATCH_SIZE, TIME_DIM, 2))
    in1 = torch.rand((BATCH_SIZE, 4))
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (BATCH_SIZE, output_shape)
    # Reset agent 0 should only clear the state of agent 0
    net.reset(0)
    assert np.all(net.hidden_state[:, 0].numpy() == 0)
    assert not np.all(net.hidden_state[:, 1].numpy() == 0)


@pytest.mark.parametrize(
    "output_sequence_network_type",
    ["mlp", "rnn"],
)
def test_fully_connected_mlp_with_bottleneck_features_and_seq_output(
    output_sequence_network_type,
):
    in0 = torch.rand((BATCH_SIZE, 2))
    in1 = torch.rand((BATCH_SIZE, 4))
    input_shapes = {"in0": in0.shape[-1:], "in1": in1.shape[-1:]}
    output_shape = 1
    output_sequence_length = 3
    net = MLPWithBottleneckFeaturesAndSequenceOutput(
        input_shapes=input_shapes,
        output_shape=output_shape,
        num_envs=1,  # Not used when input_shape n-dim == 1
        rnn_hidden_size=1,  # Not used when input_shape n-dim == 1
        num_rnn_layers=1,  # Not used when input_shape n-dim == 1
        keys_to_bottleneck=["in0"],
        bottleneck_size=5,
        norm_after_bottleneck=True,
        tanh_after_bottleneck=True,
        mlp_nodes=[16, 16],
        output_sequence_network_type=output_sequence_network_type,
        output_sequence_length=output_sequence_length,
    )
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (BATCH_SIZE, output_sequence_length, output_shape)


@pytest.mark.parametrize(
    "output_sequence_network_type",
    ["mlp", "rnn"],
)
def test_fully_connected_mlp_with_bottleneck_features_and_rnn_ins_and_seq_output(
    output_sequence_network_type,
):
    TIME_DIM = 2
    NUM_ENVS = 2
    input_shapes = {"in0": (TIME_DIM, 2), "in1": (TIME_DIM, 4)}
    output_shape = 1
    output_sequence_length = 3
    net = MLPWithBottleneckFeaturesAndSequenceOutput(
        input_shapes=input_shapes,
        output_shape=output_shape,
        num_envs=NUM_ENVS + 1,  # for eval
        rnn_hidden_size=32,
        num_rnn_layers=3,
        keys_to_bottleneck=["in0"],
        bottleneck_size=5,
        norm_after_bottleneck=True,
        tanh_after_bottleneck=True,
        mlp_nodes=[16, 16],
        output_sequence_network_type=output_sequence_network_type,
        output_sequence_length=output_sequence_length,
    )
    # When in eval mode, BATCH_SIZE must be num_envs
    net.eval()
    in0 = torch.rand((NUM_ENVS, TIME_DIM, 2))
    in1 = torch.rand((NUM_ENVS, TIME_DIM, 4))
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (NUM_ENVS, output_sequence_length, output_shape)
    # When in train mode, BATCH_SIZE can be anything
    net.train()
    in0 = torch.rand((BATCH_SIZE, TIME_DIM, 2))
    in1 = torch.rand((BATCH_SIZE, TIME_DIM, 4))
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (BATCH_SIZE, output_sequence_length, output_shape)
    # Reset agent 0 should only clear the state of agent 0
    net.reset(0)
    assert np.all(net.hidden_state[:, 0].numpy() == 0)
    assert not np.all(net.hidden_state[:, 1].numpy() == 0)


@pytest.mark.parametrize(
    "output_sequence_network_type",
    ["mlp", "rnn"],
)
def test_fully_connected_mlp_with_bottleneck_features_partial_rnn_ins_and_seq_output(
    output_sequence_network_type,
):
    TIME_DIM = 2
    NUM_ENVS = 2
    input_shapes = {"in0": (TIME_DIM, 2), "in1": (4,)}
    output_shape = 1
    output_sequence_length = 3
    net = MLPWithBottleneckFeaturesAndSequenceOutput(
        input_shapes=input_shapes,
        output_shape=output_shape,
        num_envs=NUM_ENVS + 1,  # for eval
        rnn_hidden_size=32,
        num_rnn_layers=3,
        keys_to_bottleneck=["in0"],
        bottleneck_size=5,
        norm_after_bottleneck=True,
        tanh_after_bottleneck=True,
        mlp_nodes=[16, 16],
        output_sequence_network_type=output_sequence_network_type,
        output_sequence_length=output_sequence_length,
    )
    # When in eval mode, BATCH_SIZE must be num_envs
    net.eval()
    in0 = torch.rand((NUM_ENVS, TIME_DIM, 2))
    in1 = torch.rand((NUM_ENVS, 4))
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (NUM_ENVS, output_sequence_length, output_shape)
    # When in train mode, BATCH_SIZE can be anything
    net.train()
    in0 = torch.rand((BATCH_SIZE, TIME_DIM, 2))
    in1 = torch.rand((BATCH_SIZE, 4))
    out = net({"in0": in0, "in1": in1})
    assert out.shape == (BATCH_SIZE, output_sequence_length, output_shape)
    # Reset agent 0 should only clear the state of agent 0
    net.reset(0)
    assert np.all(net.hidden_state[:, 0].numpy() == 0)
    assert not np.all(net.hidden_state[:, 1].numpy() == 0)


def test_diffusion_conditional_unet1d():
    sequence_length = 4
    action_size = 2
    actions = torch.rand((BATCH_SIZE, sequence_length, action_size))
    features = torch.rand((BATCH_SIZE, 4))
    input_shapes = {
        "actions": actions.shape[-1:],
        "features": features.shape[-1:],
        "timestep": (1,),
    }
    net = ConditionalUnet1D(
        input_shapes=input_shapes,
        output_shape=action_size,
        sequence_length=sequence_length,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    )
    out = net({"actions": actions, "features": features, "timestep": 1})
    assert out.shape == (BATCH_SIZE, sequence_length, action_size)
    assert out.shape[1:] == net.output_shape


def test_diffusion_transformer():
    sequence_length = 4
    action_size = 2
    actions = torch.rand((BATCH_SIZE, sequence_length, action_size))
    features = torch.rand((BATCH_SIZE, 4))
    input_shapes = {
        "actions": actions.shape[-1:],
        "features": features.shape[-1:],
        "timestep": (1,),
    }
    net = TransformerForDiffusion(
        input_shapes=input_shapes,
        output_shape=action_size,
        sequence_length=sequence_length,
        n_obs_steps=None,
        n_layer=12,
        n_head=12,
        n_emb=768,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=False,
        time_as_cond=True,
        n_cond_layers=0,
    )
    out = net({"actions": actions, "features": features, "timestep": 1})
    assert out.shape == (BATCH_SIZE, sequence_length, action_size)
    assert out.shape[1:] == net.output_shape


@pytest.mark.parametrize(
    "output_shape",
    [DECODER_SINGLE_CAM, DECODER_MULTI_CAM, DECODER_MULTI_CAM_AND_CH],
)
def test_decoder_cnn_multi_view(output_shape):
    input_shape = (128,)
    net = DecoderCNNMultiView(
        input_shape,
        output_shape,
        min_res=4,
        channels=32,
        kernel_size=4,
    )
    out = net(torch.rand((BATCH_SIZE,) + input_shape))
    assert out.shape == (BATCH_SIZE,) + output_shape


@pytest.mark.parametrize(
    "output_shape",
    [DECODER_SINGLE_CAM, DECODER_MULTI_CAM, DECODER_MULTI_CAM_AND_CH],
)
def test_decoder_cnn_multi_view_with_different_min_res(output_shape):
    input_shape = (128,)
    for min_res in [2, 4, 8, 16, 32]:
        net = DecoderCNNMultiView(
            input_shape,
            output_shape,
            min_res=min_res,
            channels=32,
            kernel_size=4,
        )
        out = net(torch.rand((BATCH_SIZE,) + input_shape))
        assert out.shape == (BATCH_SIZE,) + output_shape


@pytest.mark.parametrize(
    "output_shape",
    [DECODER_SINGLE_CAM, DECODER_MULTI_CAM, DECODER_MULTI_CAM_AND_CH],
)
def test_decoder_cnn_multi_view_with_different_kernels(output_shape):
    input_shape = (128,)
    for kernel_size in [3, 4, 5, 6, 7]:
        net = DecoderCNNMultiView(
            input_shape,
            output_shape,
            min_res=4,
            channels=32,
            kernel_size=kernel_size,
        )
        out = net(torch.rand((BATCH_SIZE,) + input_shape))
        assert out.shape == (BATCH_SIZE,) + output_shape
