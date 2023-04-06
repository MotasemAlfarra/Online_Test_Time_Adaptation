import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
te_transforms_inc = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])