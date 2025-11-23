"""
pseudo code for the wrapper



class ZQETrainer:
    def __init__(self, decoder, encoder, schedule):
        self.decoder = decoder
        self.encoder = encoder
        self.schedule = schedule  # α(t)

    def elbo_loss(self, y): ...

    def zq_loss(self, y): ...

    def training_step(self, y, t):
        alpha = self.schedule(t)

        # ----- encoder update (always ELBO) -----
        loss_enc = -self.elbo_loss(y)
        loss_enc.backward()
        encoder_opt.step()

        # ----- decoder update (blend) -----
        loss_dec = (1 - alpha) * (-self.elbo_loss(y)) + alpha * self.zq_loss(y)
        loss_dec.backward()
        decoder_opt.step()
"""
