# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from climax.arch import ClimaX

class RegionalClimaX(ClimaX):
    def __init__(self, default_vars, img_size=..., patch_size=2, embed_dim=1024, depth=8, decoder_depth=2, num_heads=16, mlp_ratio=4, drop_path=0.1, drop_rate=0.1):
        super().__init__(default_vars, img_size, patch_size, embed_dim, depth, decoder_depth, num_heads, mlp_ratio, drop_path, drop_rate)

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables, region_info):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        region_patch_ids = region_info['patch_ids']
        x = x[:, :, region_patch_ids, :]





        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed[:, region_patch_ids, :]

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x









    def forward(self, x, y, lead_times, variables, out_variables, metric, lat, region_info):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.
            variables: input vars, example = ["temperature", "pressure", "wind_speed", "humidity", "precipitation"] **not real id
            out_variables: output vars, vars we a predicting: example = ["wind"]  **not real id
            region_info: Containing the region's information {min_w: 56 , min_h: 23, max_w: 12, max_h: 48, min_h_pr:, min_w_pr:, max_h_pr:, max_w_pr:, patch_ids }
            metric: MSE, adjusts the loss based on the latitude: [lat_weighted_mse], list of functions(only one function)
            lat: comes from preprocessing file

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        
        #turns the data into vectors.
        out_transformers = self.forward_encoder(x, lead_times, variables, region_info)  # B, L, D'
        #makes predictions based on the data
        preds = self.head(out_transformers)  # B, L, V*p*p

        #asigns min_h
        min_h_pr, max_h_pr = region_info['min_h_pr'], region_info['max_h_pr']
        min_w_pr, max_w_pr = region_info['min_w_pr'], region_info['max_w_pr']

        #asigns min_h... etc to equal region_info min_h... etc
        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        
       # predspr = self.unpatchify(preds.clone(), h=h, w=w, min_h_pr=min_h_pr, max_h_pr=max_h_pr, min_w_pr=min_w_pr, max_w_pr=max_w_pr)
        predspr = self.unpatchify(preds.clone(), h = max_h_pr - min_h_pr + 1, w = max_w_pr - min_w_pr + 1, min_h_pr=min_h_pr, max_h_pr=max_h_pr, min_w_pr=min_w_pr, max_w_pr=max_w_pr)
        #predspr = self.unpatchify(preds.clone(), h = max_h_pr - min_h_pr + 1, w = max_w_pr - min_w_pr + 1)

        #formats the predictions to match the correct height and width
        preds = self.unpatchify(preds, h = max_h - min_h + 1, w = max_w - min_w + 1)
 
        #formats the output variables
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        #takes out only the variables specified, in our case wind
        predspr = predspr[:, out_var_ids]
        preds = preds[:, out_var_ids]

       # mask = torch.zeros_like(y, dtype=torch.float32, device=y.device)
       # mask[:, :, min_h_pr:max_h_pr + 1, min_w_pr:max_w_pr + 1] = 1
        
       # mask = mask[:, :, min_h_pr:max_h_pr + 1, min_w_pr:max_w_pr + 1]
       # lat_pr = lat[min_h_pr:max_h_pr+1]
       # y_pr = y[:, :, min_h_pr:max_h_pr + 1, min_w_pr:max_w_pr + 1]
        y_pr = y[:, :, min_h_pr:max_h_pr + 1, min_w_pr:max_w_pr + 1]
        lat_pr = lat[min_h_pr:max_h_pr + 1]

                 # Generate mask for the subsection
        mask = torch.zeros_like(y_pr, dtype=torch.float32, device=y.device)
        mask[:] = 1  # Setting the mask to 1 for the subsection region

        #print("Mask", mask, "lat_pr", lat_pr, "y_pr", y_pr)
        # if theres nothing in metric, asigns loss to none
        if metric is None:
            loss = None
        # if there is 

        #function m is the loss function(math part)
        #for each function in the list of functions metric it does the math for the loss
        else:
            loss = []
            for m in metric:
                global_loss = m(preds, y, out_variables, lat)
                pr_loss = m(predspr, y_pr, out_variables, lat_pr, mask=mask)
                #total_loss = {"loss": global_loss["loss"] + pr_loss["loss"]}
                global_loss["loss"] += pr_loss["loss"]
                loss.append(global_loss)
            #loss = [m(preds, y, out_variables, lat) + m(predspr, y_pr, out_variables, lat_pr, mask=mask) for m in metric]

        #loss is a list of integers
        return loss, preds

# Add min_h_pr to lines 87
# Add predpr variable similar to the one on line 92
# pass it through outvarids along with the global one
# add it to loss





    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix, region_info):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat, region_info=region_info)

        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]
        clim = clim[:, min_h:max_h+1, min_w:max_w+1]

        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
