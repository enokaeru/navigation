from model import QNetworklow
from torchsummary import summary

model = QNetworklow(37, 4, 0)
summary(model, input_size=(64, 37))
