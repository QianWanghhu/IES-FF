if  sampler.ntraining_samples >= num_samples[sample_step]:
                # validation_samples = sampler.training_samples[:, num_samples[sample_step - 1]:num_samples[sample_step]]
                pred_values = gp_curr(validation_samples, return_cov=False).squeeze()
                # validation_values = function(validation_samples).squeeze()

                # Compute error
                assert pred_values.shape == validation_values.shape
                error = norm(pred_values-validation_values)/norm(validation_values)
                if callback is not None:
                    callback(gp)

                print(gp.kernel_)
                print('N', ntraining_samples_curr, 'Error', error)
                errors[sample_step - 1] = error
                nsamples[sample_step -1] = ntraining_samples_curr

                sample_step += 1