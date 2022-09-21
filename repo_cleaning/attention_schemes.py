def attentions_schemes(x):
    if self.attention_type == 'space_limited':
        init_cls_token = x[:,0,:].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, 4, 1) #(2x2)
        cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=4).unsqueeze(1) #(2x2)
        xt = x[:,1:,:]
        xt = rearrange(xt, 'b (h w t) m -> b h w t m',b=B,h=H,w=W,t=T)
        xt = rearrange(xt, 'b (h1 h) (w1 w) t m -> (b h1 w1) (h w t) m',b=B,t=T, h1=2, w1=2) #h1=4, w1=5
        xt = torch.cat((cls_token, xt), 1)
        res_temporal = self.drop_path(self.attn(self.norm1(xt)))

        cls_token = res_temporal[:,0,:]
        cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=4) #(2x2)
        cls_token = torch.mean(cls_token,1,True) ## averaging for every stick

        res_temporal = res_temporal[:,1:,:]
        res_temporal = rearrange(res_temporal, '(b h1 w1) (h w t) m -> b (h1 h) (w1 w) t m',b=B,t=T, h1=2, w1=2,h=4,w=5)
        res_temporal = rearrange(res_temporal, 'b h w t m -> b (h w t) m ',b=B,h=H,w=W,t=T)

        x = x + torch.cat((cls_token, res_temporal), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    elif self.attention_type == 'time_limited':
        init_cls_token = x[:,0,:].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, 8, 1) #taking 4 slices together
        cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=8).unsqueeze(1)
        xs = x[:,1:,:]
        xs = rearrange(xs, 'b (h w t1 t) m -> (b t1) (h w t) m',b=B,h=H,w=W,t=4,t1=8)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))

        cls_token = res_spatial[:,0,:]
        cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=8)
        cls_token = torch.mean(cls_token,1,True) ## averaging for every frame

        res_spatial = res_spatial[:,1:,:]
        res_spatial = rearrange(res_spatial, '(b t1) (h w t) m -> b (h w t1 t) m',b=B,h=H,w=W,t=4,t1=8)
        res = res_spatial

        x = x + torch.cat((cls_token, res), 1) # !!! x +
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    elif self.attention_type == 'space_and_time_limited':
        iinit_cls_token = x[:,0,:].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, 16, 1) # 16 cubes
        cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=16).unsqueeze(1)
        xs = x[:,1:,:]
        xs = rearrange(xs, 'b (h1 h w1 w t1 t) m -> b (h1 h) (w1 w) (t1 t) m', b=B,h=4,w=5,t=4,h1=2,w1=2,t1=8)
        xs = rearrange(xs, 'b (h1 h) (w1 w) (t1 t) m -> (b h1 w1 t1) (h w t) m', b=B,h=4,w=5,t=4,h1=2,w1=2,t1=8)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))

        ### Taking care of CLS token
        cls_token = res_spatial[:,0,:]
        cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=16)
        cls_token = torch.mean(cls_token,1,True) ## averaging for every frame

        res_spatial = res_spatial[:,1:,:]
        res_spatial = rearrange(res_spatial, '(b h1 w1 t1) (h w t) m -> b (h1 h) (w1 w) (t1 t) m', b=B,h=4,w=5,t=4,h1=2,w1=2,t1=8)
        res_spatial = rearrange(res_spatial, 'b (h1 h) (w1 w) (t1 t) m -> b (h1 h w1 w t1 t) m', b=B,h=4,w=5,t=4,h1=2,w1=2,t1=8)
        res = res_spatial

        ## Mlp
        x = x + torch.cat((cls_token, res), 1) # !!! x +
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    elif self.attention_type == 'space_only_different_calculation':
        init_cls_token = x[:,0,:].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
        xs = x[:,1:,:]
        xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))

        ### Taking care of CLS token
        cls_token = res_spatial[:,0,:]
        cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
        cls_token = torch.mean(cls_token,1,True) ## averaging for every frame

        res_spatial = res_spatial[:,1:,:]
        res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
        res = res_spatial
        
        # x = xt

        ## Mlp
        x = x + torch.cat((cls_token, res), 1) # !!! x +
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    elif self.attention_type == 'divided_space_time':
        ## Temporal
        xt = x[:,1:,:]
        xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
        res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x[:,1:,:] + res_temporal

        ## Spatial
        init_cls_token = x[:,0,:].unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
        xs = torch.cat((cls_token, xs), 1)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))

        ### Taking care of CLS token
        cls_token = res_spatial[:,0,:]
        cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
        cls_token = torch.mean(cls_token,1,True) ## averaging for every frame

        res_spatial = res_spatial[:,1:,:]
        res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
        res = res_spatial
        
        x = xt

        ## Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x