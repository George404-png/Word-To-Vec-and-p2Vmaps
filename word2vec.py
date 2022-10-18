
class CallbackHoldoutSet(gensim.models.callbacks.CallbackAny2Vec):
    """
    Callback to compute loss on validation holdout set. Loss can be
    either negative sampling loss (negative=1, in principle faster) or
    the full softmax loss (negative=0, in principle slower). Samples
    can be either skip-gram (sg>0) or cbow (sg=0) style. The class needs
    to be initialized with a list of basket data. The class will generate
    the corresponding validation samples.
    """

    def __init__(
        self, holdout, negative=0, sg=1, random=0, log_nth_batch=100000000, verbose=0
    ):

        self.holdout = holdout
        self.context = self.center = None
        self.negative = negative  # number of negative samples
        self.sg = sg
        self.random = random
        self.losses = []
        self.losses_batch = []
        self.log_nth_batch = log_nth_batch
        self.epoch = 0
        self.epoch_batch = 0
        self.verbose = verbose
        super(CallbackHoldoutSet, self).__init__()

    def _log(self, x):
        if self.verbose > 0:
            if self.verbose > 1 or isinstance(x, str):
                logger.info(x)

    def _setup_sg_pairs(self, model):
        #
        # Construct skip-gram style center-context pairs.
        #
        # Here, the "center" word appears on the input layer and is used to predict the
        # context. All pairs of words within a certain window size of each other appear
        # as training samples.
        #

        # map tokens to index
        token_map = {k: v.index for k, v in model.wv.vocab.items()}
        all_token_indices = set([model.wv.vocab[k].index for k in model.wv.vocab])

        # loop through all sentences
        _tmp_context = []
        _tmp_center = []
        for sentence in self.holdout:
            # build all center-context pairs
            sentence = [token_map[k] for k in sentence if k in token_map]
            samples_sentence = pd.DataFrame(
                {
                    "w1": np.repeat(sentence, len(sentence)),
                    "pos1": np.repeat(np.arange(len(sentence)), len(sentence)),
                    "w2": np.tile(sentence, len(sentence)),
                    "pos2": np.tile(np.arange(len(sentence)), len(sentence)),
                }
            )

            # limit to center-context pairs within window `w`
            samples_sentence["delta_pos"] = np.abs(
                samples_sentence["pos1"] - samples_sentence["pos2"]
            )
            samples_sentence = samples_sentence[
                (samples_sentence["delta_pos"] != 0)
                & (samples_sentence["delta_pos"] <= model.window)
            ]

            # generate negative samples
            allowed_negative_samples = list(all_token_indices.difference(set(sentence)))
            negative_samples = np.random.choice(
                allowed_negative_samples, (samples_sentence.shape[0], model.negative)
            )

            # format output
            _tmp_context.append(
                samples_sentence["w1"].values.reshape(-1, 1).astype(np.int32)
            )
            _tmp_center.append(
                np.hstack(
                    [
                        samples_sentence["w2"].values.reshape(-1, 1).astype(np.int32),
                        negative_samples,
                    ]
                )
            )

        self.context = np.concatenate(_tmp_context)
        self.center = np.concatenate(_tmp_center)

    def _calculate_loss(self, model):

        # first time through, compute the training samples from sentences
        # can't do this on init, because we need model first
        if self.context is None:
            self._log("build context center pairs")
            if self.random:
                raise NotImplementedError
            elif self.sg:
                self._setup_sg_pairs(model)
            else:
                self._setup_cbow_pairs(model)
            self.n_samples = len(self.center) * (
                self.negative + 1
            )  # to get loss per sample

        loss = self._calculate_loss_vectorized(model)

        return loss

    def _calculate_loss_vectorized(self, model):

        # Efficiently compute loss using vectorized numpy operations.
        l1 = (
            np.einsum("ijk->ik", model.wv.vectors[self.context]) / self.context.shape[1]
        )
        if self.center.shape[1] == 1:
            # softmax
            raise NotImplementedError
        else:  # negative sampling
            l2 = model.trainables.syn1neg[self.center]
            scores = -np.einsum("ik,ijk->ij", l1, l2)
            scores[:, 0] *= -1
            scores = np.float64(scores)
            loss = -np.sum(np.log(scipy.special.expit(scores)))

        return loss

    def on_batch_end(self, model):
        if self.epoch_batch != 0 and self.epoch_batch % self.log_nth_batch == 0:
            self._log("batch callback -- %d" % self.epoch_batch)
            batch_end_loss = self._calculate_loss(model)
            self.losses_batch.append(batch_end_loss)
        self.epoch_batch += 1
        return

    def on_epoch_begin(self, model):
        if self.epoch == 0:
            self.losses.append(self._calculate_loss(model))
        self._log("epoch callback -- %d" % self.epoch)
        self.epoch_batch = 0
        return

    def on_epoch_end(self, model):
        epoch_end_loss = self._calculate_loss(model)
        self.losses.append(epoch_end_loss)
        self.epoch += 1
        return

