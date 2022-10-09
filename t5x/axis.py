axis = {
  decoder: {
    cross_position_k_linear: {
      kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
    },
    cross_position_q_linear: {
      kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
    },
    decoder_norm: {
      scale_axes: AxisMetadata(names=('embed',)),
    },
    layers_0: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_1: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_10: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_11: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_12: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_13: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_14: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_15: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_16: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_17: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_18: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_19: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_2: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_20: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_21: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_22: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_23: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_3: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_4: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_5: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_6: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_7: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_8: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    layers_9: {
      encoder_decoder_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_cross_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_self_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      self_attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
    },
    pe_pre_ln: {
      scale_axes: AxisMetadata(names=('embed',)),
    },
    position_embedding: {
      embedding_axes: AxisMetadata(names=('vocab', 'embed')),
    },
    pre_ln: {
      scale_axes: AxisMetadata(names=('embed',)),
    },
    segments_embedding: {
      embedding_axes: AxisMetadata(names=('vocab', 'embed')),
    },
    self_position_k_linear: {
      kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
    },
    self_position_q_linear: {
      kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
    },
  },
  encoder: {
    encoder_norm: {
      scale_axes: AxisMetadata(names=('embed',)),
    },
    layers_0: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_1: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_10: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_11: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_12: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_13: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_14: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_15: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_16: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_17: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_18: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_19: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_2: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_20: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_21: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_22: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_23: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_3: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_4: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_5: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_6: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_7: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_8: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    layers_9: {
      attention: {
        key: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        out: {
          kernel_axes: AxisMetadata(names=('joined_kv', 'embed')),
        },
        query: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
        value: {
          kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
        },
      },
      mlp: {
        wi_0: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wi_1: {
          kernel_axes: AxisMetadata(names=('embed', 'mlp')),
        },
        wo: {
          kernel_axes: AxisMetadata(names=('mlp', 'embed')),
        },
      },
      pre_attention_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
      pre_mlp_layer_norm: {
        scale_axes: AxisMetadata(names=('embed',)),
      },
    },
    pe_pre_ln: {
      scale_axes: AxisMetadata(names=('embed',)),
    },
    position_k_linear: {
      kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
    },
    position_q_linear: {
      kernel_axes: AxisMetadata(names=('embed', 'joined_kv')),
    },
  },
  token_embedder: {
    embedding_axes: AxisMetadata(names=('vocab', 'embed')),
  },
}