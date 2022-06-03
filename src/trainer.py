from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy as CCELoss
from tensorflow.keras.losses import BinaryCrossentropy as BCELoss
from tensorflow.keras.optimizers import Adam

from src.model import Backbone, Discriminator


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)

    def custom_grad(dy):
        return -dy

    return y, custom_grad


class Trainer(tf.keras.Model):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.logdir = args.logdir
        self.writer = tf.summary.create_file_writer(self.logdir)
        self.eval_every = args.eval_every
        self.epochs_stage_1 = args.epochs_stage_1
        self.epochs_stage_2 = args.epochs_stage_2

        self.model_1 = Backbone()
        self.model_2 = Backbone()
        self.discriminator = Discriminator()

        self.lr = args.lr
        self.optimizer_pre = Adam(self.lr)
        self.optimizer_gen = Adam(self.lr)
        self.optimizer_disc = Adam(self.lr)
        self.optimizer_gen_and_clf = Adam(self.lr)

        self.reverse_gradients = args.reverse_gradients

    def train_stage_1(self, train_dataset, eval_dataset):
        self.model_1.trainable = True
        self.model_2.trainable = False
        self.discriminator.trainable = False

        self.model_1.compile(optimizer=self.optimizer_pre, loss=CCELoss())
        self.model_1.fit(
            train_dataset,
            validation_data=eval_dataset,
            epochs=self.epochs_stage_1,
        )

    def train_stage_2(self, train_dataset, eval_dataset):
        self.model_1.trainable = False
        self.model_2.trainable = True
        self.discriminator.trainable = True

        def train_one_batch(x_source, y_source, x_target, step, training):
            with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc, tf.GradientTape() as tape_gen_and_clf:
                # feature extraction
                f_source_pretrained = self.model_1.f(x_source, training=False)
                f_source_aligned = self.model_2.f(x_source, training=training)
                f_target = self.model_2.f(x_target, training=training)

                # step b: classification loss
                y_pred = self.model_2.clf(f_source_aligned, training=training)
                L_clf = CCELoss()(y_source, y_pred)

                # step c: consistency loss
                L_c = tf.reduce_mean(
                    tf.abs(f_source_pretrained - f_source_aligned), axis=-1
                )

                # step d: reverse gradient and adversarial alignment loss
                if self.reverse_gradients:
                    f_source_aligned = grad_reverse(f_source_aligned)
                    f_target = grad_reverse(f_target)
                    loss_sign = -1
                else:
                    loss_sign = 1

                D_source_pretrained = self.discriminator(
                    f_source_pretrained, training=training
                )
                D_target = self.discriminator(f_target, training=training)

                L_d_disc = BCELoss()(
                    tf.ones_like(D_source_pretrained), D_source_pretrained
                ) + BCELoss()(tf.zeros_like(D_target), D_target)
                L_d_gen = BCELoss()(tf.zeros_like(D_target), D_target)

                # step e: overall loss and update
                loss_gen_and_clf = tf.reduce_mean(L_clf)
                loss_gen = tf.reduce_mean(-loss_sign * L_d_gen + L_c)
                loss_disc = tf.reduce_mean(loss_sign * L_d_disc)

                with self.writer.as_default(step=step):
                    tag = "train" if training else "eval"
                    tf.summary.scalar(f"{tag}/L_clf", tf.reduce_mean(L_clf))
                    tf.summary.scalar(f"{tag}/L_c", tf.reduce_mean(L_c))
                    tf.summary.scalar(f"{tag}/L_d", tf.reduce_mean(L_d_gen))
                    tf.summary.scalar(
                        f"{tag}/p(Y=source | x_source)",
                        tf.reduce_mean(D_source_pretrained),
                    )
                    tf.summary.scalar(
                        f"{tag}/p(Y=source | x_target)",
                        tf.reduce_mean(D_target),
                    )

            if training:
                weights_gen_and_clf = self.model_2.trainable_variables
                gradients_gen_and_clf = tape_gen_and_clf.gradient(
                    loss_gen_and_clf, weights_gen_and_clf
                )
                self.optimizer_gen_and_clf.apply_gradients(
                    zip(gradients_gen_and_clf, weights_gen_and_clf)
                )

                weights_gen = self.model_2.f.trainable_variables
                gradients_gen = tape_gen.gradient(loss_gen, weights_gen)
                self.optimizer_gen.apply_gradients(
                    zip(gradients_gen, weights_gen)
                )

                weights_disc = self.discriminator.trainable_variables
                gradients_disc = tape_disc.gradient(loss_disc, weights_disc)
                self.optimizer_disc.apply_gradients(
                    zip(gradients_disc, weights_disc)
                )

        train_batch_count = 0
        eval_batch_count = 0
        for epoch in tqdm(
            range(self.epochs_stage_2), desc="epochs", leave=True
        ):
            for batch in tqdm(train_dataset, desc="train batches", leave=False):
                x_source, y_source, x_target = batch
                train_one_batch(
                    x_source,
                    y_source,
                    x_target,
                    step=train_batch_count,
                    training=True,
                )
                train_batch_count += 1

            if epoch % self.eval_every == 0:
                for step, batch in tqdm(
                    enumerate(eval_dataset), desc="eval batches", leave=False
                ):
                    x_source, y_source, x_target = batch
                    train_one_batch(
                        x_source,
                        y_source,
                        x_target,
                        step=eval_batch_count,
                        training=False,
                    )
                    eval_batch_count += 1
