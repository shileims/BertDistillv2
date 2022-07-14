

def get_num_layer_for_swin(var_name, num_max_layer, depth):
    if var_name.startswith("vision_model.backbone.patch_embed"):
        return 0
    elif var_name.startswith("vision_model.backbone.layers"):
        var_name = var_name.replace("vision_model.backbone.layers", "layers")
        layer_id = int(var_name.split('.')[1])
        block_id = var_name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm' or block_id == 'channel_reduction':
            return sum(depth[:layer_id + 1])
        layer_id = sum(depth[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_max_layer - 1

def get_num_layer_for_unilm(var_name, num_max_layer):
    if var_name.startswith("language_model.backbone.embeddings") or var_name.startswith("language_model.backbone.encoder.rel_pos_bias"):
        return 0
    elif var_name.startswith("language_model.backbone.encoder.layer"):
        var_name = var_name.replace("language_model.backbone.encoder.layer", "layer")
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner(object):
    def __init__(self, values, depth=None, is_sentence=False):
        self.values = values
        self.depth = depth
        self.is_sentence = is_sentence

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        if self.is_sentence:
            return get_num_layer_for_unilm(var_name, len(self.values))
        else:
            return get_num_layer_for_swin(var_name, len(self.values), self.depth)