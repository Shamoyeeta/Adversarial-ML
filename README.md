# Adversarial-ML

In artificial intelligence (AI) and machine learning, deep learning models represent a new paradigm of learning. With the rising interest in Deep Neural Networks owing to it’s large scope in solving everyday problems, there has also been major concern about how robust these models are. In this thesis we looked into some of the well-known attacks and tried to understand the principles behind them. In addition, we examined some of the defence strategies that might be employed to ward against these attacks.

One consideration that we wanted to focus on was to find a defence that required minimal changes in existing convolutional networks. Hence, we implement the DeepLDA model which replaces the categorical cross entropy layer of the model with Linear Discriminant Analysis objective function. We then compare the accuracy of adversarial examples on DeepLDA model as compared to existing model with categorical cross entropy function.

The research has been conducted and written as a part of Computer Science and Engineering curriculum in Motilal Nehru National Institute of Technology, Allahabad.
