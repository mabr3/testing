import numpy as np
import matplotlib.pyplot as plt
from matplotlib import container
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from math import ceil
import math
import sys
#sns.set()

class Graphs:
    def __init__(self):
        print("ok")
    

    def plot_mse(control_data, preds, pre_data, preds_pre, post_data,preds_post, _vars, mean, std, NR_VISITS):
        mse=[]
        for i in range(control_data.shape[0]):
            s=0
            for k in range(len(_vars)):
                for j in range(NR_VISITS):
                    s+= math.pow((control_data[i,j,k]-preds[i,j,k]),2)
                    
            #mse.append((1.0/control_data.shape[0])*s)    
            mse.append(s)
        for i in range(pre_data.shape[0]):
            s=0
            for k in range(len(_vars)):
                for j in range(NR_VISITS):
                    s+= math.pow((pre_data[i,j,k]-preds_pre[i,j,k]),2)
                    
            #mse.append((1.0/pre_data.shape[0])*s)
            mse.append(s)
        
        for i in range(post_data.shape[0]):
            s=0
            for k in range(len(_vars)):
                for j in range(NR_VISITS):
                    s+= math.pow((post_data[i,j,k]-preds_post[i,j,k]),2)
                    
            #mse.append((1.0/post_data.shape[0])*s)
            mse.append(s)

        r = np.arange(0,control_data.shape[0] + pre_data.shape[0] + post_data.shape[0],1)
        Y = ['control'] *control_data.shape[0]  + ['pre'] * pre_data.shape[0] + ['post'] * post_data.shape[0]
        fig = plt.figure(1,figsize=(20,7))
        #sns.scatter(r,mse,color=Y,s=10)
        muted    = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
        newPal   = dict(control = muted[0], pre = muted[4], post = muted[2])
        sns.scatterplot(r, mse, hue=Y,palette=newPal)
        plt.show()
        return plt
        
        
    
    # Plots after pred #
    def plot_loss(batch,intermediate,latent,beta, history):
        fig = plt.figure(1, figsize=(14,7))
        plt.plot(range(1, len(history.history['loss'])+1), history.history['loss'], label='Training Loss')
        plt.plot(range(1, len(history.history['val_loss'])+1), history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title("batch size %d, intermediate_dim %d, , latent_dim %d, beta %f" % (batch,intermediate,latent,beta))
        start, end = plt.ylim()
        plt.yticks(np.arange(0, round(end), end/10.0))
        plt.grid(linestyle='--', alpha=0.7)
        print("\n Min -> ", min(history.history['loss']))
        plt.show()
        return (min(history.history['loss']), min(history.history['val_loss'])), plt
    
    # Plot some info on the available data. For now only a histogram of cag distribution
    def generate_profile_plots(persons):
        hist_control=[]
        hist_pre = []
        hist_post = []
        
        for i in persons:
            j=int(i.caghigh)
            if i.state ==2:
                hist_pre.append(j)
            elif i.state ==3:
                hist_post.append(j)
            elif i.state in [4,5]:
                hist_control.append(j)
    
        fig1, axs1 = plt.subplots(1,3, figsize=(15, 8), sharex='col')
        axs1[0].hist(hist_control, bins=[i for i in range(10,72)], facecolor='blue', alpha=0.75, rwidth=0.5)
        axs1[1].hist(hist_pre, bins=[i for i in range(10,72)], facecolor='yellow', alpha=0.75, rwidth=0.5)
        axs1[2].hist(hist_post, bins=[i for i in range(10,72)], facecolor='red', alpha=0.75, rwidth=0.5)
        axs1[0].yaxis.grid()
        axs1[1].yaxis.grid()
        axs1[2].yaxis.grid()
        return
        
    # Plots after prediction
        
    def generate_pred_plots(control_data, preds, pre_data, preds_pre, post_data,preds_post, _vars, mean, std, NR_VISITS):
        
        control_data *= std
        control_data += mean
        preds *= std
        preds += mean
    
        pre_data *= std
        pre_data += mean
        preds_pre *= std
        preds_pre += mean
    
        post_data *= std
        post_data += mean
        preds_post *= std
        preds_post += mean
        
        
        n_plots = NR_VISITS
        #print(n_plots)
        n_cols = n_plots // 2
        #print(n_cols)
        n_rows = n_plots // n_cols
        #print(n_rows)
        n_rows += n_plots % n_cols
        #print(n_rows)
        position = np.arange(1, n_plots +1)
        
        data = np.concatenate((control_data,pre_data,post_data))
        data_preds = np.concatenate((preds, preds_pre,preds_post))
        Y = ['blue'] *control_data.shape[0]  + ['yellow'] * pre_data.shape[0] + ['red'] * post_data.shape[0]
        r = np.arange(0,control_data.shape[0] + pre_data.shape[0] + post_data.shape[0],1)
        
    
        for var in _vars:
            _index = _vars.index(var)
            fig, ax = plt.subplots(NR_VISITS,1, figsize=(18,5*NR_VISITS))
            fig.suptitle(var, fontsize=20)
            fig.tight_layout()
            for i in range(NR_VISITS):
                ax[i].scatter(r,data[:,i,_index], label ='data', facecolors='none', edgecolor=Y, s=15)
                ax[i].scatter(r,data_preds[:,i,_index], label ='predictions', facecolors='none', edgecolor='black', s=15)
                ax[i].legend(loc='upper right')
                ax[i].set_title("Visit %d" % (i+1))

            plt.grid(linestyle='--', alpha=0.7)
            plt.show()
        return
    
    
        # Plots from the chosen variables and only on the used data.
    
    def generate_vars_plots(_vars,mean, mean_pre, mean_post, std, std_pre, std_post, control_data, pre_data, post_data,NR_VISITS):
    
        def autolabel():
            # Attach a text label next to each value to display its value (only used for mean atm)
            
            for i in ind:
                axs1[0].text(i + 0.24*width, mean[i, _index] + axs1[0].get_yaxis().get_tick_padding()*0.2, '%f' % mean[i, _index], fontsize=15)
                axs1[0].text(i + 1.04*width, mean_pre[i, _index] + axs1[0].get_yaxis().get_tick_padding()*0.2, '%f' % mean_pre[i, _index], fontsize=15)
                axs1[0].text(i + 1.84*width, mean_post[i, _index] + axs1[0].get_yaxis().get_tick_padding()*0.2, '%f' % mean_post[i, _index], fontsize=15)
    
        for var in _vars:
            _index = _vars.index(var)
            ind = np.arange(NR_VISITS)
            width = 0.30
            fig1, axs1 = plt.subplots(2, 1, figsize=(14, 7), sharex='col')
            fig1.suptitle(var, fontsize=16)
            
            rects_1 = axs1[0].errorbar(ind + 0.2*width, mean[:, _index], std[:, _index], color='b', linestyle='None', marker='o', capsize=2, label="Control")
            rects_2 = axs1[0].errorbar(ind + width, mean_pre[:, _index], std_pre[:, _index], color='y', linestyle='None', marker='o', capsize=2, label="Pre")
            rects_3 = axs1[0].errorbar(ind + (1.8*width), mean_post[:, _index], std_post[:, _index], color='r',
                                       linestyle='None', marker='o', capsize=2, label="Post")
            
            axs1[0].set_ylabel('Value')
            axs1[0].set_title("Mean and std")
            axs1[0].set_xticks(ind + width)
            try:
                _, end = axs1[0].get_ylim()
                start=0
            except:
                start=0
                end=4
                
            axs1[0].set_yticks(np.arange(round(start), round(end)+1, (end-start)/5.00))
            axs1[0].grid(linestyle='--', alpha=0.7)
            autolabel()
    
            # Remove std drawing from the legend
            handles, labels = axs1[0].get_legend_handles_labels()
            new_handles = []
    
            for h in handles:
                # only need to edit the errorbar legend entries
                if isinstance(h, container.ErrorbarContainer):
                    new_handles.append(h[0])
                else:
                    new_handles.append(h)
    
            fig1.legend(new_handles, labels, loc='center right')
            fig1.subplots_adjust(hspace=0.15)
    
            # Scatter Plot #
            
            axs1[1].set_ylabel('Value')
            axs1[1].set_xlabel('Visit Number')
            axs1[1].set_title("Scatter plot")
            axs1[1].set_xticks(ind + width)
            axs1[1].set_xticklabels(np.arange(1,NR_VISITS+1))
    
            # create x values for all the Y values with random to distribute them along the visit's width
            for z in ind:
                _x1 = np.ones(control_data.shape[0])*(z + 0.2*width) + (np.random.uniform(-1.0, 1.0, control_data.shape[0]) * width/3)
                axs1[1].scatter(_x1, control_data[:, z, _index], facecolors='none', edgecolor='b', s=5)
                
                _x2 = np.ones(pre_data.shape[0])*(z + width) + (np.random.uniform(-1.0, 1.0, pre_data.shape[0]) * width/3)
                axs1[1].scatter(_x2, pre_data[:, z, _index], facecolors='none', edgecolor='y', s=5)
                
                _x3 = np.ones(post_data.shape[0])*(z + 1.8*width) + (np.random.uniform(-1.0, 1.0, post_data.shape[0]) * width/3)
                axs1[1].scatter(_x3, post_data[:, z, _index], facecolors='none', edgecolor='r', s=5)
            
            try:
                _, end = axs1[1].get_ylim()
                start=0
            except ValueError:
                print(sys.exc_info()[0])
                start=0
                end=4
            
            axs1[1].set_yticks(np.arange(round(start)-1, round(end)+1, round((end-start)/5.0)))
            axs1[1].grid(linestyle='--', alpha=0.7)
        
        plt.show()
        return
    
    
    def plot_PCA(enc, control_data, pre_data, post_data,control_labels, pre_labels, post_labels, persons,subjids):
        pca = PCA(n_components=2)

        data = np.concatenate((control_data,pre_data, post_data))
        labels = np.concatenate((control_labels, pre_labels, post_labels))
        l,_  = enc.predict(data)

        z = pca.fit_transform(l)
        
        Y = ['control'] *control_data.shape[0]  + ['pre'] * pre_data.shape[0] + ['post'] * post_data.shape[0]

        muted    = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
        newPal   = dict(control = muted[0], pre = muted[4], post = muted[2])
        _style = []
        
        for ind,i in enumerate(labels):
            _style.append(persons[int(i)].sex)
        f=plt.figure(figsize=(20,10))
        #print(_style)
        plt.grid()
        sns.scatterplot(z[:,0], z[:,1], hue=Y,palette=newPal, style=_style)
        

        plt.title("PCA")
        plt.show()
        return plt
    
    def explain_PCA(enc, control_data, pre_data, post_data,control_labels, pre_labels, post_labels, persons, n_components):
        
        pca = PCA(n_components=n_components)
        l0,_ = enc.predict(control_data)
        l1,_ = enc.predict(pre_data)
        l2,_ = enc.predict(post_data)
        
        l= np.concatenate((l0,l1,l2))

        z = pca.fit_transform(l)
        
        print("Explained variance ratio:")
        print(pca.explained_variance_ratio_)
        f0=plt.figure(figsize=(20,10))
        plt.grid()
        sns.lineplot(data=pca.explained_variance_ratio_)
        plt.title("Explained Variance Ratio")
        plt.show()
        return plt
        
        
    
    def plot_TSNE(enc, control_data, pre_data, post_data, control_labels, pre_labels, post_labels, persons):
        #l0,_ = enc.predict(control_data)
        #l1,_ = enc.predict(pre_data)
        #l2,_ = enc.predict(post_data)
        #l= np.concatenate((l0,l1,l2))
        data = np.concatenate((control_data,pre_data, post_data))
        labels = np.concatenate((control_labels, pre_labels, post_labels))
        l,_  = enc.predict(data)

        fig1, axs1 = plt.subplots(1, 3, figsize=(20, 7))
        Y = ['control'] *control_data.shape[0]  + ['pre'] * pre_data.shape[0] + ['post'] * post_data.shape[0]
        muted    = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
        newPal   = dict(control = muted[0], pre = muted[4], post = muted[2])

        for a, perp  in enumerate([25,50,100]):
            tsne = TSNE(n_components=2, random_state=0, n_iter=2500, perplexity=perp)
            z = tsne.fit_transform(l)
            sns.scatterplot(z[:,0], z[:,1], hue=Y, ax=axs1[a],palette=newPal)
            axs1[a].grid()
            axs1[a].set_title(perp)

        fig1.suptitle("T-SNE")
        plt.show()
        return plt
    