"""
Tests sklearn linear classifiers converter.
"""
import unittest

import numpy as np
import torch
from skl2pytorch import convert_sklearn
from skl2pytorch.common.data_types import Float32TensorType
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC, NuSVC


class TestSklearnLinearClassifiers(unittest.TestCase):
    def test_logistic_regression_multi(self):
        model = LogisticRegression(solver="liblinear", multi_class="ovr", fit_intercept=True)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(
            model.predict_proba(X), pytorch_model(torch.from_numpy(X))[1].data.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_logistic_regression(self):
        """
        TODO: this may have a bug
        """
        model = LogisticRegression(solver="liblinear", fit_intercept=True)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(
            model.predict_proba(X), pytorch_model(torch.from_numpy(X))[1].data.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_linear_regression(self):
        model = LinearRegression()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([100, 200]))])

        self.assertTrue(pytorch_model is not None)
        sklearn = model.predict(X)
        pytorch = pytorch_model(torch.from_numpy(X)).data.numpy().flatten()
        np.testing.assert_allclose(sklearn, pytorch, rtol=1e-5, atol=1e-6)

    def test_logistic_regression_cv(self):
        model = LogisticRegressionCV(solver="liblinear", multi_class="ovr", fit_intercept=True)
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(
            model.predict_proba(X), pytorch_model(torch.from_numpy(X))[1].data.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_linear_svc(self):
        """
        TODO: this may have a bug.
        """
        model = LinearSVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)
        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_linear_svc_multi(self):
        model = LinearSVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_svc(self):
        model = SVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)
        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_svc_multi(self):
        model = SVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_nu_svc(self):
        model = NuSVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)
        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-4, atol=1e-6)

    def test_nu_svc_multi(self):
        model = NuSVC()
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)
        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])

        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_sgd_classifier(self):
        """
        TODO: this may have a bug
        """
        model = SGDClassifier(loss="log")
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(2, size=100)

        model.fit(X, y)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)

    def test_sgd_classifier_multi(self):
        model = SGDClassifier(loss="log")
        X = np.random.rand(100, 200)
        X = np.array(X, dtype=np.float32)
        y = np.random.randint(3, size=100)

        model.fit(X, y)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([1, 20]))])
        self.assertTrue(pytorch_model is not None)
        np.testing.assert_allclose(model.predict(X), pytorch_model(torch.from_numpy(X))[0].data.numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
