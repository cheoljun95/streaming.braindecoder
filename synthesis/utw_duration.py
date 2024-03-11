import torch
import numpy as np
import soundfile as sf
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import logging
############################
########## Models ###########

class PastFCEncoder(torch.nn.Module):
    '''
    Autoregressive class in CARGAN
    https://github.com/descriptinc/cargan/blob/master/cargan/model/condition.py#L6
    '''
    def __init__(self, input_len=512, hidden_dim=256, output_dim=128):
        '''
        Args:
            input_len: the number of samples of autoregressive conditioning
        '''
        super().__init__()

        model = [
            torch.nn.Linear(input_len, hidden_dim),
            torch.nn.LeakyReLU(.1)]
        for _ in range(3):
            model.extend([
                torch.nn.Linear(
                    hidden_dim,
                    hidden_dim),
                torch.nn.LeakyReLU(.1)])
        model.append(
            torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*model)
    
    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, out_channels, ar_input/out_channels)
                eg (batch_size, 1, input_len) for waveforms

        Return:
            shape (batch_size, output_dim)
        '''
        # (B, C, T)
        # (B, 1, T)
        x = x.reshape(x.shape[0], -1)
        return self.model(x)
    

class ResidualBlock(torch.nn.Module):
    """Residual block module in HiFiGAN."""

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """Initialize HiFiGANResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = torch.nn.ModuleList()
        if use_additional_convs:
            self.convs2 = torch.nn.ModuleList()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        for dilation in dilations:
            self.convs1 += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        bias=bias,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            ]
            if use_additional_convs:
                self.convs2 += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x
    
class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, nout, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )


class DurPredictor(torch.nn.Module):
    def __init__(
        self, idim, num_dur_layers=2, dur_channels=384, kernel_size=3, dropout=0.1, shift=1.0
    ):
        super(DurPredictor, self).__init__()
        self.shift = shift
        self.conv = torch.nn.ModuleList()
        for idx in range(num_dur_layers):
            idur_channels = idim if idx == 0 else dur_channels
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        idur_channels,
                        dur_channels,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(dur_channels, dim=1),
                    torch.nn.Dropout(dropout),
                )
            ]
        self.linear = torch.nn.Linear(dur_channels, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        # (B, C, T)
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        
        # (B, T)
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)
        
        # NOTE: log domain
        if is_inference:
            # NOTE: linear domain
            # avoid negative value
            xs = torch.clamp(
                torch.round(xs.exp() - self.shift), min=0
            ).long()  
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def forward(self, xs, x_masks=None):
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        return self._forward(xs, x_masks, True)


class LenChanger(torch.nn.Module):
    def __init__(self, pad_value=0.0):
        super().__init__()
        self.pad_value = pad_value

    def _pad(self, elems, pad):
        batch_size = len(elems)
        elems_padded = elems[0].new(batch_size, max(x.size(0) for x in elems), *elems[0].size()[1:]).fill_(pad)
        for i in range(batch_size):
            elems_padded[i, : elems[i].size(0)] = elems[i]
        return elems_padded

    def forward(self, xs, ds, alpha=1.0):
        if alpha != 1.0:
            assert alpha > 0
            ds = torch.round(ds.float() * alpha).long()
        if ds.sum() == 0:
            ds[ds.sum(dim=1).eq(0)] = 1
        repeat = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)]
        return self._pad(repeat, self.pad_value)


class HiFiGANGeneratorDur(torch.nn.Module):
    """Discrete Symbol HiFiGAN generator with duration predictor module."""

    def __init__(self, in_channels=80, out_channels=1, channels=512, kernel_size=7,
        upsample_scales=(8, 8, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4),
        paddings=None, output_paddings=None,
        resblock_kernel_sizes=(3, 7, 11), resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True, bias=True, nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1}, use_weight_norm=True,
        use_ar=False, ar_input=512, ar_hidden=256, ar_output=128, use_tanh=True,
        use_spk_id=False, num_spk=None, use_spk_emb=False, spk_emb_size=32, spk_emb_hidden=32,
        use_ph=False, num_ph=None, ph_emb_size=8, use_aux_loss=False, aux_dim=None,
        use_emb=False, num_emb=512, emb_dim=1024, emb_p=None,
        num_dur_layers=2, dur_channels=384, dur_kernel_size=3, dur_shift=1.0, dur_dropout=0.5,
    ):
        super().__init__()

        self.use_ar = use_ar
        self.use_spk_id = use_spk_id
        self.use_spk_emb = use_spk_emb
        self.use_ph = use_ph
        self.use_aux_loss = use_aux_loss
        self.use_emb = use_emb

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)
        
        if paddings is None:
            paddings = [upsample_scales[i] // 2 + upsample_scales[i] % 2 for i in range(len(upsample_kernel_sizes))]
        else:
            new_paddings = []
            for i, s in enumerate(paddings):
                if s == "default":
                    new_paddings.append(upsample_scales[i] // 2 + upsample_scales[i] % 2)
                else:
                    print("not implemented")
                    exit()
            paddings = new_paddings
        if output_paddings is None:
            output_paddings = [upsample_scales[i] % 2 for i in range(len(upsample_kernel_sizes))]
        else:
            new_output_paddings = []
            for i, s in enumerate(output_paddings):
                if s == "default":
                    new_output_paddings.append(upsample_scales[i] % 2)
                else:
                    print("not implemented")
                    exit()
            output_paddings = new_output_paddings

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            # assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=paddings[i],
                        output_padding=output_paddings[i],
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        if use_tanh:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        else:
            self.output_conv = torch.nn.Sequential(
                # NOTE(kan-bayashi): follow official implementation but why
                #   using different slope parameter here? (0.1 vs. 0.01)
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    1,
                    padding=(kernel_size - 1) // 2,
                ),
            )

        
        if use_ar:
            self.ar_model = PastFCEncoder(input_len=ar_input, hidden_dim=ar_hidden, output_dim=ar_output)
        if use_spk_id:
            assert num_spk is not None
            self.spk_emb_mat = torch.nn.Embedding(num_spk, spk_emb_size)
            self.spk_fc = torch.nn.Linear(spk_emb_size, in_channels)
        if use_spk_emb:
            self.spk_fc = torch.nn.Linear(spk_emb_size, spk_emb_hidden)
        if use_ph:
            assert num_ph is not None
            self.ph_emb_mat = torch.nn.Embedding(num_ph, ph_emb_size)
        if use_aux_loss:
            final_scale = np.prod(upsample_scales)
            self.aux_fc = torch.nn.Linear(channels // (2 ** (i + 1)), aux_dim)
            ph_pooling = "AvgPool1d"
            assert final_scale % 2 == 0
            ph_pooling_params = {"kernel_size": final_scale*2, "stride": final_scale, "padding": final_scale//2}
            self.aux_pool = getattr(torch.nn, ph_pooling)(**ph_pooling_params)
        if use_emb:
            if emb_p is None:
                self.emb_mat = torch.nn.Embedding(num_emb, emb_dim)
            else:
                try:
                    init_array = np.load(emb_p)
                    
                    # Infer pad_id from init array.
                    pad_id = init_array.shape[0]
                    self.pad_id = pad_id
                    init_array = np.concatenate((init_array, np.zeros((1, init_array.shape[1])) ), axis = 0)
                    self.emb_mat = torch.nn.Embedding.from_pretrained(torch.tensor(init_array).float(), freeze=False, padding_idx=pad_id)
                except:
                    logging.info(f"Can't load emb_mat from {emb_p}. Using default initialization instead.")
                    self.emb_mat = torch.nn.Embedding(num_emb, emb_dim)
        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        self.dur_predictor = DurPredictor(
            in_channels,
            num_dur_layers=num_dur_layers,
            dur_channels=dur_channels,
            kernel_size=dur_kernel_size,
            dropout=dur_dropout,
            shift=dur_shift,
        )

        self.len_changer = LenChanger()

    def forward(self, c, spk_id=None, spk=None, ar=None, ph=None, dur=None):
        if self.use_emb:
            # (B, T)
            mask = c == self.pad_id
            c = c.long()
            c = self.emb_mat(c)  # (batchsize, seq_len, emb_dim)
            c = c.transpose(1, 2)  # (batchsize, emb_dim, seq_len)
        # logging.info(('after emb', c.shape))
        if self.use_ar:
            ar_feats = self.ar_model(ar) # (batchsize, ar_output)
            ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, c.shape[2]) # (batchsize, ar_output, length)
            c = torch.cat((c, ar_feats), dim=1)
        # logging.info(('after ar', c.shape))
        if self.use_spk_emb:
            cspk = self.spk_fc(spk)
            cspk = cspk.unsqueeze(2).repeat(1, 1, c.shape[2])
            c = torch.cat((c, cspk), dim=1)
        if self.use_spk_id:
            spk_emb = self.spk_emb_mat(spk_id)  # (batchsize, spk_emb_size)
            spk_emb = self.spk_fc(spk_emb)  # (batchsize, in_channels)
            spk_emb = spk_emb.unsqueeze(2).repeat(1, 1, c.shape[2])  # (batchsize, in_channels, length)
            c = c + spk_emb
        if self.use_ph:
            ph_feats = self.ph_emb_mat(ph)  # (batchsize, length, ph_emb_size)
            ph_feats = ph_feats.transpose(1, 2)
            c = torch.cat((c, ph_feats), dim=1)
        ds_out = self.dur_predictor(c.transpose(1, 2), x_masks = mask)
        c = self.len_changer(c.transpose(1, 2), dur).transpose(1, 2)
        c = self.input_conv(c)

        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks  # (batch_size, some_channels, length)
        # logging.info(c.shape)
        out = self.output_conv(c)  # (batch_size, 1, input_len*final_scale)
        # logging.info(out.shape)
        if self.use_aux_loss:
            c = c.transpose(1, 2)
            ph_out = self.aux_fc(c)
            ph_out = ph_out.transpose(1, 2)
            ph_out = self.aux_pool(ph_out)
            return (out, ph_out)
        else:
            return out, ds_out

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")

    # TODO
    def inference(self, c, normalize_before=False, ar=None, spk=None, dur=None):
        assert len(c.shape) == 2, f"Input must have shape (1, T) but got {c.shape} and must be long!"
        
        # Embedding.
        # (1, T) => (1, T, C) => (1, C, T)
        c = self.emb_mat(c)  
        c = c.transpose(1, 2)
        
        # AR features.
        ar_feats = self.ar_model(ar) # (batchsize, ar_output)
        ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, c.shape[2]) # (batchsize, ar_output, length)
        c = torch.cat((c, ar_feats), dim=1)
       
        # Duration modelling.
        ds_out = self.dur_predictor.inference(c.transpose(1, 2))
        
        # Change the input shape based on duration output.
        c = self.len_changer(c.transpose(1, 2), ds_out).transpose(1, 2)
        c = self.input_conv(c)

        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks  # (batch_size, some_channels, length)
        # logging.info(c.shape)
        out = self.output_conv(c)  # (batch_size, 1, input_len*final_scale)
        return out, ds_out
    
############################
########### Utils ##########

def get_fid(file):
    return os.path.basename(file).split(".")[0]

def load_config(config_dir):
    with open(config_dir) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config 

def load_model(checkpoint, config=None, stats=None, generator2=False, strict=True):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.

    Return:
        torch.nn.Module: Model instance.

    """
    if generator2:
        type_key = "generator2_type"
        params_key = "generator2_params"
        generator_key = "generator2"
    else:
        type_key = "generator_type"
        params_key = "generator_params"
        generator_key = "generator"
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
            
    # workaround for typo #295
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config[params_key].items()
    }
    model = HiFiGANGeneratorDur(**generator_params)
    load_output = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(
        load_output["model"][generator_key],
    )
    return model

def load_model_eval(model_dir, device = torch.device("cuda:0"), return_config = False):
    if model_dir[-4:] == ".pkl":
        model_dir = os.path.dirname(model_dir)
    config_file = os.path.join(model_dir, "config.yml")
    ckpt = os.path.join(model_dir, "best_mel_ckpt.pkl")
    
    config = load_config(config_file)
    model = load_model(ckpt, config).to(device)
    print(f"Loaded model parameters from {ckpt}.")
    model.remove_weight_norm()
    model = model.eval()
    if return_config:
        return model, config
    return model

#################################
########### Data Utils ##########
def abbr2key(s):
    if s == 'h':
        return 'hubert'
    if s == 'hls':
        return 'hubert_ls960_ft_2'
    elif s == 'hd':
        return 'hubert_160'
    elif s == 'a':
        return 'art'
    elif s == 'm':
        return 'mel'
    elif s == 'w':
        return 'audio'
    elif s == 'tv':
        return 'tv9_scaled'
    elif s == 'tvpm':
        return 'tvpmf8nema'
    elif s == 'tvpffemad':
        return 'tvpffema_160'
    elif s == 'pffemad':
        return 'pffema_160'
    elif s == 'tvpfd':
        return 'tvpf_160'
    elif s == 'tvpffd':
        return 'tvpff_160'
    elif s == 'tvfd':
        return 'tvf_160'
    elif s == 'tvd':
        return 'tv9_scaled_160'
    return s


def get_feat_type(s):
    if s == 'audio' or s == 'lj_vits_scaled' or s == 'lj_vits_scaled_16' or s == 'lj_vits_scaled_20' or s == 'lj_vits_ph_scaled_20' \
            or s.startswith('wav_') or s == 'wav' or s == 'flac' or s == "clean_flac" or s=="all_wav_16k_vf_napa_voice":
        return 'audio'
    else:
        return 'feat'


def get_task_features(dataset_mode):
    """
    Args:
        dataset_mode: str, e.g., "a2w"

    Return:
        x_key: str, input feature
        y_key: str, output feature
        feature_key: str, non-audio feature
            unused in package_mode == 'window'
        feature_dict: keys are features used in tasks (strings)
            values are types ('audio' or 'feat')
    """
    if '2' in dataset_mode:
        mlist = dataset_mode.split('2')
        if len(mlist) == 3:
            mlist = [mlist[0], '2'.join(mlist[1:])]
        x_key = abbr2key(mlist[0])
        y_key = abbr2key(mlist[1])
        if get_feat_type(y_key) != 'audio' and y_key != 'ph':
            feature_key = y_key
        else:
            assert get_feat_type(x_key) != 'audio' and x_key != 'ph'
            feature_key = x_key
        feature_dict = {x_key: get_feat_type(x_key), y_key: get_feat_type(y_key)}
    else:
        feature_dict = {dataset_mode: get_feat_type(dataset_mode)}
        x_key = dataset_mode
        y_key = dataset_mode
        feature_key = None
    return x_key, y_key, feature_key, feature_dict

def ar_loop(model, x, config, normalize_before=True, use_reduction = True,
            modality=None, spk=None, audio_chunk_len=None):
    '''
    Args:
        x: (art_len, num_feats)
    
    Return:
        signal: (audio_len,)
    '''
    x_key, y_key, feature_key, feature_dict = get_task_features(config["dataset_mode"])
    params_key = "generator_params"
    in_2d = not feature_dict[x_key] == 'audio'  # any([config["dataset_mode"].startswith(mode) for mode in mode_in_2ds])
    out_2d = not feature_dict[y_key] == 'audio'  # any([config["dataset_mode"].endswith(mode) for mode in mode_out_2ds])
    if audio_chunk_len is None:
        audio_chunk_len = config["batch_max_steps"]
        
    # Note: This if statement will be true.
    if in_2d:
        # In_chunk_len = audio_chunk_len / 320 = number of input units per chunk.
        in_chunk_len = int(audio_chunk_len/config["hop_size"])
    else:
        in_chunk_len = audio_chunk_len
    if out_2d:
        out_chunk_len = int(audio_chunk_len/config["hop_size"])
        past_out_len = int(config[params_key]["ar_input"]/config["generator_params"].get("ar_channels", config["generator_params"]["out_channels"]))
    else:
        out_chunk_len = audio_chunk_len
        past_out_len = config[params_key]["ar_input"]
        
    # print(in_chunk_len)
    # Break the input into chunks. 
    ins = [x[i:i+in_chunk_len] for i in range(0, len(x), in_chunk_len)]
    if not in_2d and len(ins[-1]) < config["hop_size"]:
        ins = ins[:-1]
        
    # (1, 1, 512). Stores the preceding synthesized speech.
    prev_samples = torch.zeros((1, config["generator_params"].get("ar_channels", config["generator_params"]["out_channels"]), past_out_len), dtype=torch.float, device=x.device)
    outs = []
    for cin in ins:
        # print("prev samples: ", prev_samples.shape)
        # cin: (T,)
        if use_reduction:
            cin, dur = torch.unique_consecutive(
                cin.clone().long(), return_counts=True, dim=0
            )
        # Model forward inference.
        # cin: (1, T)
        # print("cin shape: ", cin.shape)
        cin = cin.unsqueeze(0)
        
        # cout: (1, 1, T)
        cout, dur_pred = model.inference(cin, normalize_before=normalize_before, ar=prev_samples, spk=spk)
        # print("cout", cout.shape, "dur", dur_pred.shape, "expected wav len: ", cin.shape[-1] * 320)
        outs.append(cout[0][0])
        
        if cout.shape[-1] > prev_samples.shape[-1]:
            prev_samples = cout[:, :, -prev_samples.shape[-1]:]
        else:
            prev_samples[:, :, -cout.shape[-1]:] = cout
        # if past_out_len <= out_chunk_len:
        #     prev_samples = cout[:, :, -past_out_len:]
        # else:
        #     prev_samples[:, :, :-out_chunk_len] = prev_samples[:, :, out_chunk_len:].clone()
        #     prev_samples[:, :, -out_chunk_len:] = cout
            
            
    out = torch.cat(outs, dim=0)  # w2a (seq_len, num_feats)
    return out

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

################################################
########### Hificar Speech synthesis ###########
# input_file: the file contains a list of units.
# output_dir: output folder for the synthesized speech.
def hificar_speech_synthesis(input_file, output_dir, ckpt1, 
                             gpu_number = 0, output_sr = 16000):
    mkdir(output_dir)
    # Load the model.
    device = torch.device(f"cuda:{gpu_number}")
    print(f"ckpt1:\n{ckpt1}")
    # device = torch.device("cpu")
    model, config = load_model_eval(ckpt1, device, return_config=True)
    model.eval()
    
    # Inference
    with torch.no_grad():
        # Load the file.
        c = np.load(input_file)
        c = torch.tensor(c, dtype=torch.long).to(device)
        
        # Inference speech.
        # 640 is for extreme cases... but still intelligible.
        # IMPORTANT: if the RNNT output is not in collapsed form, set use_reduction = True
        # else: set use_reduction = False.
        y = ar_loop(model, c, config, normalize_before=False, 
                    use_reduction=True,
                    audio_chunk_len=640) # was 60000
        
        # Save the speech.
        sf.write(os.path.join(output_dir, get_fid(input_file)+'.wav'), y.cpu().numpy(), output_sr)
        
# file: scp file path. 
# returns (list of fids, list of file paths)
def read_scp(file):
    with open(file, 'r') as inf:
        lines = inf.readlines()
        lines = [l.strip() for l in lines]
        fids = []
        featps = []
        for l in lines:
            l_list = l.split()
            fid = l_list[0]
            featp = l_list[1]
            fids.append(fid)
            featps.append(featp)
    
    return fids, featps
'''
# hub_units_6-100 
fids, files = read_scp("./atsn/egs/ema/data/test_vctk.scp")
input_files = []
for i in files:
    fid = get_fid(i)
    file = os.path.join(os.path.dirname(os.path.dirname(i)), "hub_units_6-100", fid + ".npy")
    input_files.append(file)
    break


# input_file: a npy file containing the units.
# output_dir: change the save directory. 
# ckpt1: the trained hificar model checkpoint
# gpu_number: which gpu to put the model on.


# Three ckpts:
# 1. /data/vocoder_ckpts/lj_hub_6-100_units2wav_duration/best_mel_ckpt.pkl
# 2. /data/vocoder_ckpts/vctk_hub_6-100_units2wav_duration/best_mel_ckpt.pkl
# 3. /data/vocoder_ckpts/vctk_lj_hub_6-100_units2wav_duration/best_mel_ckpt.pkl
hificar_speech_synthesis(input_file=input_files[0],
                        output_dir = './test',
                        ckpt1="/data/vocoder_ckpts/lj_hub_6-100_units2wav_duration/best_mel_ckpt.pkl",
                        gpu_number=0)
                        
'''