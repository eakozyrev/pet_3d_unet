import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
#python compare_weights.py models_promts_atten_norm_scatt_random_emb/Model_0.h5 models_promts_atten_norm_scatt_random_emb/Model_1.h5


if __name__=='__main__':

    model0 = keras.models.load_model(sys.argv[1], compile=False)
    print(model0.layers)

    model1 = keras.models.load_model(sys.argv[2], compile=False)

    for i in range(0,len(model0.layers)):
        try:
            conv0 = model0.layers[i].get_weights()[0]
            conv1 = model1.layers[i].get_weights()[0]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle(f'{i} layer')
            p1 = np.sum(conv0,axis=(0,1,2))
            p2 = np.sum(conv1,axis=(0,1,2))
            ax1.imshow(p1)
            ax1.set_title('Model1')
            ax2.imshow(p2)
            ax2.set_title('Model2')
            ax3.imshow((p1 - p2)/(p1 + 0.001))
            ax3.set_title('1 - Model2/Model1')
            plt.show()
        except: pass
