# Aries_proj
# Neural Style Transfer with VGG19

##  Overview

This project implements Neural Style Transfer (NST) using a pre-trained VGG19 network. The objective of NST is to blend the content of one image with the style of another, creating a new image that retains the core elements of the content image while adopting the artistic features of the style image. This technique leverages the power of deep learning and convolutional neural networks (CNNs) to extract and recombine visual features.

To achieve this, following methods was employed:

1. Loss Function-based NST: This method creates a loss function ensuring similarity between the generated image and both the content and style images using VGG19.

---

## Installation Instructions

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- Matplotlib
- TensorFlow version: 2.15.0

### Setup Environment

1. **Clone the repository:**
    ```sh
   git clone https://github.com/yourusername/Neural-Style-Transfer.git
   cd Neural-Style-Transfer
   ```

---

## Usage
See the image below you can correspondingly choose from wide range of content images and wide range of style we have multiple categories too,so accordingly modify path and image.

![Demo](https://drive.google.com/uc?export=view&id=1ZI5cVkEASdjU_zPfzQ7ZSgIBnw95ggVS)

```sh
StylePath1 = '../input/best-artworks-of-all-time/images/images/Jan_van_Eyck/'
ContentPath1 = '../input/image-classification/validation/validation/food/'
content_imagi1 = image_loader(ContentPath1+'0Bc8QvgJwOInun6Ibxo4.jpeg')
content_size1 = get_image_size(content_imagi1)
style_imagi1 = image_loader(StylePath1+'Jan_van_Eyck_1.jpg',size = content_size1)
global_fun(content_imagi1,style_imagi1)
```

## Dependencies

The project requires the following libraries:

- torch
- torchvision
- pillow
- matplotlib

You can install them using the following command:
```sh
pip install torch torchvision pillow matplotlib tensorflow
```

---
## Directory Str

```
Neural-Style-Transfer/
│
├── content/
│   └── your_content_image.jpg
├── style/
│   └── your_style_image.jpg
├── output/
│   └── your_output_image.jpg
│
├── nueral_style.ipynb
├── requirements.txt
└── README.md
```

---

## Method
1. Start with checking you have the latest version of tensorflow
   ```sh
   import tensorflow as tf
   print("TensorFlow version:", tf.__version__)
   ```

2. Load and process images modify their sized based on computation power.
 ```sh
# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

print(imsize)

# scale imported image
# transform it into a torch tensor
loader = transforms.Compose([transforms.Resize(imsize),  transforms.ToTensor()])
```
3. Create functions for normalizing the image. loading it, pre-viewing it.
4. Content and style loss: The content loss measures the difference between the content image and the generated image in terms of high-level features, while the style loss measures the difference in the style of the images using Gram matrices(statistical values)
   ```sh
   class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

    # This is for the style loss
    def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

    # Same structure as the content loss
    class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    ```
5. Importing vgg19 model and using it's desired depth layers to compute style/content losses:
    ```sh
    # Importing the VGG 19 model like in the paper (here we set it to evaluation mode)
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # VGG network are normalized with special values for the mean and std
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # create a module to normalize input image so we can easily put it in a nn.Sequential
    class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
    # Here we insert the loss layer at the right spot
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    ```
6 & 7. Defining the main model and optimization:
```sh
      def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# This will run the neural style transfer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] >= 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    
    input_img.data.clamp_(0, 1)

    return input_img
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_image, style_image, input_img, num_steps=200, style_weight = 1000000)
```

## End
I have created a separate function at the end which can be runned by just providing the proper inputs.

 ## Results
 ![Demo](https://drive.google.com/uc?export=view&id=1GzytNiRw8UmcKDUU9q4cJAQaLKdcpx5p)
 ![Demo](https://drive.google.com/uc?export=view&id=10jqff7lphFBYUBQz_7wLOqBLGuASZXvb)
 ![Demo](https://drive.google.com/uc?export=view&id=1XNSIBCmWSff7UpGIj8c1H9Xf9QzCbWhf)
 ![Demo](https://drive.google.com/uc?export=view&id=1QFMaHlYzYjgQ6DDd9vVrTD1YcLr3SJKW)
 



