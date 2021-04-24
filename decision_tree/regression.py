import numpy as np
import pandas as pd


class TreeNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature      # index of the feature (x) to split on
        self.threshold = threshold  # threshold (s) of the feature: (x < s) vs (x >= s)
        self.value = value          # leaf value for end node
        self.left = left            # left branch for (x < s)
        self.right = right          # right branch for (x >= s)


class RegressionTree:
    def __init__(self, X=None, y=None, depth=6, verbose=True):
        self.depth = depth
        self.root = None
        self.pandas_features = None
        if X is not None:
            self.fit(X, y, verbose)

    @staticmethod
    def _loss_split_i(i, n, cumsum_y, cumsum_y2):
        """the left and right loss if we split at i"""
        loss_0 = cumsum_y2[i] - cumsum_y[i]**2 / i if i > 0 else 0.
        loss_1 = (
            (cumsum_y2[-1] - cumsum_y2[i]) - (cumsum_y[-1] - cumsum_y[i])**2 / (n - i)
            if n - i > 0 else 0.
        )
        return loss_0, loss_1

    @staticmethod
    def _get_best_split(x, y):
        """Find the best split point for x into R0 = {x: x < s} and R1 = {x: x >= s}, and
        Args:
            x (list or np.array): feature
            y (list or np.array): target
        Returns:
            Tuple(loss, the split s, y_R0, y_R1, indices for R0, indices for R1, loss_0, loss_1)

        Here, we find the best split of x such that it minimize the mean square error
            loss = sum_{i in R0}{ (y_i - y_R0)^2 } + sum_{i in R1}{ (y_i - y_R1)^2 },
        where y_i is the true lable and y_R is the predicted value for the region R
        Since for each region, the best y_R = mean_{i in R}(y_i), we can use cumsum to
        get the mean of y_i in O(1) time (after building the cumsum, which takes O(N)). Then since
        sum_i{ (y_i - y_R)^2 } = sum_i{y_i^2} - 2 * sum_i{y_i * y_R} + sum_i{y_R^2}
                               = sum_i{y_i^2} - 2 * sum_i{y_i} * y_R + N * y_R^2,
                               = sum_i{y_i^2} - sum_i{y_i}^2 / N,
        we can compute the total loss also in O(N) once the cumsum of y^2 is built.
        """
        xyi = sorted((xi, yi, i) for i, (xi, yi) in enumerate(zip(x, y)))
        cumsum_y = [0]
        cumsum_y2 = [0]
        for _, y, _ in xyi:
            cumsum_y.append(y + cumsum_y[-1])
            cumsum_y2.append(y * y + cumsum_y2[-1])
        n = len(xyi)
        best_i = 0
        best_loss_0, best_loss_1 = RegressionTree._loss_split_i(best_i, n, cumsum_y, cumsum_y2)
        best_loss = best_loss_0 + best_loss_1
        prev_x = xyi[0][0]
        for i in range(1, n):
            x, y, _ = xyi[i]
            if x == prev_x:
                continue
            prev_x = x
            loss_0, loss_1 = RegressionTree._loss_split_i(i, n, cumsum_y, cumsum_y2)
            loss = loss_0 + loss_1
            if loss < best_loss:
                best_i = i
                best_loss = loss
                best_loss_0, best_loss_1 = loss_0, loss_1
        s = xyi[i][0]  # split point
        y_R0 = cumsum_y[i] / i if i > 0 else 0.        # left leaf value
        y_R1 = (cumsum_y[-1] - cumsum_y[i]) / (n - i)  # right leaf value
        R0 = [xyi[i][2] for i in range(best_i)]
        R1 = [xyi[i][2] for i in range(best_i, n)]
        return best_loss, s, y_R0, y_R1, R0, R1, best_loss_0, best_loss_1

    def build_node(self, X, y, y_R, loss, level):
        if level > self.depth or len(X) <= 1:
            return TreeNode(value=y_R), loss
        best_result = float('inf'), None  # best_loss, ...
        best_feat = 0
        for feat in range(self.n_features):
            result = self._get_best_split(X[:, feat], y)
            if result[0] < best_result[0]:
                best_result = result
                best_feat = feat
        loss, s, y_R0, y_R1, R0, R1, loss_0, loss_1 = best_result

        node = TreeNode(feature=best_feat, threshold=s)
        X0, y0 = X[R0], y[R0]
        X1, y1 = X[R1], y[R1]
        node.left, loss_0 = self.build_node(X0, y0, y_R0, loss_0, level + 1)
        node.right, loss_1 = self.build_node(X1, y1, y_R1, loss_1, level + 1)
        return node, loss_0 + loss_1

    def fit(self, X, y, verbose=True):
        """Fit a regression tree

        Args:
            X (pd.Dataframe or 2D np.ndarray): features matrix
            y (pd.Series or 1D np.ndarray): targets
        """
        if len(X) != len(y):
            raise ValueError("X and y lengh don't match")
        if len(X) < 1:
            raise ValueError("X and y need to have non-zero length")

        # get features columns:
        if isinstance(X, pd.DataFrame):
            self.pandas_features = X.columns
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError("X needs to be 2D")
        if len(y.shape) != 1:
            raise ValueError("X needs to be 1D")
        self.features = range(len(X[0]))
        self.n_features = len(self.features)

        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(y.shape) != 1:
            raise ValueError("y needs to be 1D")

        # building decision tree
        self.root, loss = self.build_node(X, y, y_R=y[0], loss=0, level=1)
        loss /= len(X)
        if verbose:
            print('loss (MSE):', loss)

    def predict_row(self, x_row):
        node = self.root
        while node.feature is not None:
            if x_row[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        """Predict for features X
        Args:
            X (pd.Dataframe or 2D np.ndarray): feature matrix
        Return:
            prediction (1D np.ndarray)
        """
        if self.root is None:
            raise ValueError("model hasn't been trained yet.")
        if isinstance(X, pd.DataFrame):
            if self.pandas_features is not None:
                X = X.loc[:, self.pandas_features]
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError("X needs to be 2D")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X need to have {self.n_features} features but got {X.shape[1]}"
            )
        y = np.array([self.predict_row(x_row) for x_row in X])
        return y

    def evaluate(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y lengh don't match")
        if len(X) < 1:
            raise ValueError("X and y need to have non-zero length")
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(y.shape) != 1:
            raise ValueError("y needs to be 1D")
        y_pred = self.predict(X)
        return np.mean((y_pred - y)**2)


if __name__ == '__main__':
    model = RegressionTree(depth=8)
    X = np.random.random_sample((1000, 10))
    y = np.random.random_sample(1000) * 5
    print('Training the regression decision tree...')
    model.fit(X, y)
    print('Finished training.')
    print('Evaluation loss (MSE):', model.evaluate(X, y))
