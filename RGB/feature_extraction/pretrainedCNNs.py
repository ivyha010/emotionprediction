import torch
import torchvision.models as models
from torch import optim, nn

# Remove the last fc layr to extract 2048 (or 4069)-feature vector
class FeatureExtraction_ResNet18(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(FeatureExtraction_ResNet18, self).__init__()
        pretrained_model = models.resnet18(pretrained=True)
        for param in pretrained_model.parameters():  # freeze all parameters
            param.requires_grad = False

        modules = list(pretrained_model.children())[:-1]  # delete the last fc layer

        self.modified_pretrained = nn.Sequential(*modules)

        self.bn = nn.BatchNorm1d(512)

    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.modified_pretrained(images)
        ftrs = ftrs.reshape(ftrs.size(0), -1)
        ftrs = self.bn(ftrs)
        return ftrs


class FeatureExtraction_ResNet34(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-34 and replace top fc layer."""
        super(FeatureExtraction_ResNet34, self).__init__()
        pretrained_model = models.resnet34(pretrained=True)
        for param in pretrained_model.parameters():  # freeze all parameters
            param.requires_grad = False

        modules = list(pretrained_model.children())[:-1]  # delete the last fc layer

        self.modified_pretrained = nn.Sequential(*modules)

        self.bn = nn.BatchNorm1d(512)

    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.modified_pretrained(images)
        ftrs = ftrs.reshape(ftrs.size(0), -1)
        ftrs = self.bn(ftrs)
        return ftrs


class FeatureExtraction_ResNet50(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(FeatureExtraction_ResNet50, self).__init__()
        pretrained_model = models.resnet50(pretrained=True)
        for param in pretrained_model.parameters():  # freeze all parameters
            param.requires_grad = False

        modules = list(pretrained_model.children())[:-1]  # delete the last fc layer

        self.modified_pretrained = nn.Sequential(*modules)

        #self.bn = nn.BatchNorm1d(2048)

    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.modified_pretrained(images)
        ftrs = ftrs.reshape(ftrs.size(0), -1)
        #ftrs = self.bn(ftrs)
        return ftrs


class FeatureExtraction_ResNet101(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super(FeatureExtraction_ResNet101, self).__init__()
        pretrained_model = models.resnet101(pretrained=True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        modules = list(pretrained_model.children())[:-1]  # delete the last fc layer.

        self.modified_pretrained = nn.Sequential(*modules)

        self.bn = nn.BatchNorm1d(2048)

    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.modified_pretrained(images)   # ftrs means features
        ftrs = ftrs.reshape(ftrs.size(0), -1)  # Extract a 2048-feature vector
        ftrs = self.bn(ftrs)
        return ftrs


class FeatureExtraction_ResNet152(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(FeatureExtraction_ResNet152, self).__init__()
        pretrained_model = models.resnet152(pretrained=True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        modules = list(pretrained_model.children())[:-1]  # delete the last fc layer.

        self.modified_pretrained = nn.Sequential(*modules)

        self.bn = nn.BatchNorm1d(2048)

    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.modified_pretrained(images)   # ftrs means features
        ftrs = ftrs.reshape(ftrs.size(0), -1)  # Extract a 2048-feature vector
        ftrs = self.bn(ftrs)
        return ftrs

class FeatureExtraction_VGG19(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(FeatureExtraction_VGG19, self).__init__()
        pretrained_model = models.vgg19(pretrained = True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.features = pretrained_model.features

        # Convert all the layes to list and remove the last one
        modules = list(pretrained_model.classifier.children())[:-1]  # delete the last linear layer.

        # Convert it into container and add it to our model class
        self.modified_pretrained = nn.Sequential(*modules)

        self.bn = nn.BatchNorm1d(4096)


    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.features(images)
        ftrs = ftrs.reshape(ftrs.size(0), -1)
        ftrs = self.modified_pretrained(ftrs)
        ftrs = self.bn(ftrs)
        return ftrs


class FeatureExtraction_VGG16(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(FeatureExtraction_VGG16, self).__init__()
        pretrained_model = models.vgg16(pretrained = True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.features = pretrained_model.features

        # Convert all the layes to list and remove the last one
        modules = list(pretrained_model.classifier.children())[:-1]  # delete the last fc layer.

        # Convert it into container and add it to our model class
        self.modified_pretrained = nn.Sequential(*modules)

        self.bn = nn.BatchNorm1d(4096)

    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.features(images)
        ftrs = ftrs.reshape(ftrs.size(0), -1)
        ftrs = self.modified_pretrained(ftrs)
        ftrs = self.bn(ftrs)
        return ftrs


class FeatureExtraction_AlexNet(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(FeatureExtraction_AlexNet, self).__init__()
        pretrained_model = models.alexnet(pretrained = True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.features = pretrained_model.features

        # Convert all the layes to list and remove the last one
        modules = list(pretrained_model.classifier.children())[:-1]  # delete the last fc layer.

        # Convert it into container and add it to our model class
        self.modified_pretrained = nn.Sequential(*modules)

        self.bn = nn.BatchNorm1d(4096)


    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.features(images)   
        ftrs = ftrs.reshape(ftrs.size(0), -1)
        ftrs = self.modified_pretrained(ftrs)
        ftrs = self.bn(ftrs)
        return ftrs


