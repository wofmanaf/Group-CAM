class BaseCAM(object):
    def __init__(self, model, target_layer="module.layer4.2"):
        super(BaseCAM, self).__init__()
        self.model = model.eval()
        self.gradients = dict()
        self.activations = dict()

        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, x, class_idx=None, retain_graph=False):
        raise NotImplementedError

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)