from typing import Dict
import torch
from torch import nn
from transformer4planning.models.utils import *
from transformer4planning.models.encoder.base import TrajectoryEncoder



class CNNDownSamplingResNet(nn.Module):
    def __init__(self, d_embed, in_channels, resnet_type='resnet18', pretrain=False):
        super(CNNDownSamplingResNet, self).__init__()
        self.d_embed = d_embed
        import torchvision.models as models
        if resnet_type == 'resnet18':
            self.cnn = models.resnet18(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 512
        elif resnet_type == 'resnet34':
            self.cnn = models.resnet34(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 512
        elif resnet_type == 'resnet50':
            self.cnn = models.resnet50(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 2048
        elif resnet_type == 'resnet101':
            self.cnn = models.resnet101(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 2048
        elif resnet_type == 'resnet152':
            self.cnn = models.resnet152(pretrained=pretrain, num_classes=d_embed)
            cls_feature_dim = 2048
        else:
            assert False, f'Unknown resnet type: {resnet_type}'
        self.cnn = torch.nn.Sequential(*(list(self.cnn.children())[1:-1]))
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=cls_feature_dim, out_features=d_embed, bias=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.cnn(x)
        output = self.classifier(x.squeeze(-1).squeeze(-1))
        return output


from transformers.activations import ACT2FN
class STRMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        projector_hidden_act = 'gelu'
        vision_hidden_size = 768
        model_hidden_size = config.get("d_embed")
        self.d_embed = model_hidden_size
        self.linear_1 = nn.Linear(vision_hidden_size, model_hidden_size, bias=True)
        self.act = ACT2FN[projector_hidden_act]
        self.linear_2 = nn.Linear(model_hidden_size, model_hidden_size, bias=True)

    def forward(self, image_features):
        # print("image feature shape",image_features.shape)
        # print("model hidden size",self.d_embed)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states




class STREncoder(nn.Module):
    def __init__(self,feature_dim, config):
        super().__init__()
        self.config = config
        action_kwargs = dict(
            d_embed=self.config.n_embd
        )

        cnn_kwargs = dict(
            d_embed=self.config.n_embd ,
            in_channels=self.config.raster_channels,
            resnet_type=self.config.resnet_type,
            pretrain=self.config.pretrain_encoder
        )
        self.cnn_downsample = CNNDownSamplingResNet(d_embed=cnn_kwargs.get("d_embed", None),
                                            in_channels=cnn_kwargs.get("in_channels", None),
                                            resnet_type=cnn_kwargs.get("resnet_type", "resnet18"),
                                            pretrain=cnn_kwargs.get("pretrain", False))
        print('Building ResNet encoder')
        self.action_m_embed = nn.Sequential(nn.Linear(3, action_kwargs.get("d_embed")), nn.Tanh())
        self.image_processor = None
        self.camera_image_encoder = None
        self.image_feature_connector = None
        if config.camera_image_encoder == 'dinov2':
            # WIP
            from transformers import AutoImageProcessor, Dinov2Model
            try:
                print("downloading dino")
                self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                self.camera_image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
            except:
                # using local checkpoints due to the GFW blocking of China
                print("loading local dino")
                self.image_processor = AutoImageProcessor.from_pretrained("/public/MARS/t4p/dinov2", local_files_only=True)
                self.camera_image_encoder = Dinov2Model.from_pretrained("/public/MARS/t4p/dinov2", local_files_only=True)
            # self.camera_image_m_embed = nn.Sequential(nn.Linear(257*768, action_kwargs.get("d_embed")), nn.Tanh())
            # self.camera_image_m_embed = nn.Sequential(nn.Linear(768, action_kwargs.get("d_embed")), nn.Tanh())
            # self.camera_image_m_embed = nn.Sequential(nn.Linear(768, action_kwargs.get("d_embed"), bias=False))
            self.camera_image_m_embed = STRMultiModalProjector(action_kwargs)
            # 将模型参数转换为float32
            self.camera_image_encoder = self.camera_image_encoder.float()
            

            # 确保嵌入层也转换为float32
            self.camera_image_m_embed = STRMultiModalProjector(action_kwargs).float()
                    
            for param in self.camera_image_encoder.parameters():
                param.requires_grad = False
        # there are two encoders for camera image, one is the image processor, the other is the encoder
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Nuplan raster encoder require inputs:
        `high_res_raster`: torch.Tensor, shape (batch_size, 224, 224, seq)
        `low_res_raster`: torch.Tensor, shape (batch_size, 224, 224, seq)
        `context_actions`: torch.Tensor, shape (batch_size, seq, 4 / 6)
        `trajectory_label`: torch.Tensor, shape (batch_size, seq, 2/4), depend on whether pred yaw value
        `pred_length`: int, the length of prediction trajectory
        `context_length`: int, the length of context actions

        To use camera image encoder, the input should also contain:
        `camera_image`: torch.Tensor, shape (batch_size, 8(cameras), 1080, 1920, 3)
        """
        
        camera_feature: torch.Tensor = features["camera_feature"] # (B*T, 3, 256, 1024)
        lidar_feature: torch.Tensor = features["lidar_feature"] # (B*T, 1, 256, 256)
        status_feature: torch.Tensor = features["status_feature"] # (B, 8)
        past_feature: torch.Tensor = features["past_trajectory"] # (B, T, 3)
        B = past_feature.shape[0]
        T = camera_feature.shape[1]
        past_feature = past_feature.view([B*T,-1])
        context_length = T*2
        
        res_embed = self.cnn_downsample(lidar_feature.view(B*T, 1, 256, 256))
        res_embed = res_embed.view(B, T, -1)
        device = res_embed.device


        pred_length = self.config.n_action_steps

        action_embed = self.action_m_embed(past_feature)
        action_embed = action_embed.view(B, T, -1)
        action_seq_length = T*2
        input_embeds = torch.zeros(B, action_seq_length, self.config.n_embd).to(device)
        input_embeds[:,::2,:]=res_embed
        input_embeds[:,1::2,:]=action_embed
        
        
        if self.camera_image_encoder is not None:
            
            camera_image = camera_feature.view(B,-1, 256, 1024,3)
            camera_image= camera_image[:,-1,...]
            camera_image = camera_image.squeeze()
            camera_image = self.image_processor(camera_image, return_tensors="pt").to(device)
            camera_image = self.camera_image_encoder(**camera_image).last_hidden_state
            camera_image = self.camera_image_m_embed(camera_image).to(device)
            input_embeds = torch.cat([input_embeds, camera_image], dim=1)
            context_length +=257
        print("input_embeds shape",input_embeds.shape)
        info_dict = {
            "pred_length": pred_length,
            "context_length": context_length,
        }
        
        
        return input_embeds, info_dict