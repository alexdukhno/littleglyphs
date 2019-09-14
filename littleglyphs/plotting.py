import numpy as np

import matplotlib
import matplotlib.pyplot as plt

    
# --- Visualisation utilities ---    

def visualize_glyph_list(glyph_list, N_glyphs_to_show = 50, imgsize=16, blur_factor=0.5, figsize=(12,3)):

    rasters = glyph_list.render(
        imgsize, 
        blur=True, blur_factor=blur_factor,
        up_down_sample=True, normalize=True, mode='set'
    )
    
    fig_rows = (N_glyphs_to_show // 10)
    if (N_glyphs_to_show % 10) != 0:
        fig_rows += 1
    fig_cols = 10
    fig,axs = plt.subplots(fig_rows,fig_cols,figsize=figsize)
    for i, ax in enumerate(axs.flat):
        if i<len(rasters):
            ax.imshow(rasters[i], cmap='Greys')
            label = str(int(rasters.categories[i]))
            ax.text(1, 2, label, bbox={'facecolor': 'white', 'pad': 2})
        #ax.axis('off')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
    return fig,axs

    

    
def visualize_img_array(img_array, img_categories, N_images_to_show = 10, figsize=(12,6), **options):
    show_probabilities = options.get('show_probabilities')
    probabilities = options.get('probabilities')
    if show_probabilities:
        border_cmap = matplotlib.cm.get_cmap(options.get('cmap'), 10)
    
    fig_rows = int(np.ceil(np.sqrt(N_images_to_show/2)))
    fig_cols = int(np.ceil(np.sqrt(N_images_to_show/2))) * 2
    fig,axs = plt.subplots(fig_rows,fig_cols,figsize=figsize)
    for i, ax in enumerate(axs.flat):
        if i<len(img_array):
            ax.imshow(img_array[i], cmap='Greys')
            label = ''
            if img_categories is not None:            
                label = str(int(img_categories[i]))
            if show_probabilities:
                label += ': ' + str(int(probabilities[i]*100)) + '%'
                for spine in ax.spines.values():
                    spine.set_edgecolor(
                        border_cmap(probabilities[i])  
                    )
                    spine.set_linewidth(2)
            if img_categories is not None:
                ax.text(1, 2, label, bbox={'facecolor': 'white', 'pad': 2})
            
        #ax.axis('off')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
    return fig,axs



