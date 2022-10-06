import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from PIL import Image

centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = datasets.make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
plt.scatter(X[:,0],X[:,1],s=10, alpha=0.8)
plt.title("Initial Frame")

images = []

plt.savefig('Frames/frame_initial.jpeg')
im_frame = Image.open('Frames/frame_initial.jpeg')
np_frame = np.array(im_frame)
images.append(np_frame)

class DBC():

    def __init__(self, dataset, min_pts, epsilon):
        self.dataset = dataset
        self.min_pts = min_pts
        self.epsilon = epsilon

        self.colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        self.colors = np.hstack([self.colors] * 20)

    def eps_neighborhood(self,P):
        neighborhood = []
        for Pn in range(len(self.dataset)):
            if np.linalg.norm(self.dataset[P] - self.dataset[Pn]) <=self.epsilon:
                neighborhood.append(Pn)
        return neighborhood
    
    def create_cluster_from(self, P, assignments, label):
        assignments[P] = label
        
        neighborhood = self.eps_neighborhood(P)
        
        while neighborhood:
            next_P = neighborhood.pop()
            
            if assignments[next_P] == label:
                continue
            
            assignments[next_P] = label
            
            if len(self.eps_neighborhood(next_P)) >= self.min_pts:
                # We have another core point!
                neighborhood += self.eps_neighborhood(next_P)
                
            
        
        return assignments
    
    def dbscan(self):
        """
        returns a list of assignments. The index of the
        assignment should match the index of the data point
        in the dataset.
        """

        images = []
        
        assignments = [0 for _ in range(len(self.dataset))]
        label = 1
        
        for P in range(len(self.dataset)):
            if assignments[P]!=0:
                 continue
            if len(self.eps_neighborhood(P)) >= self.min_pts:
                # we have found a core point
                assignments = self.create_cluster_from(P, assignments, label)
                label+=1

            plt.scatter(self.dataset[:, 0], self.dataset[:, 1], color=self.colors[assignments].tolist(), s=10, alpha=0.8)
            plt.title("Frame "+str(len(images)))

            plt.savefig('Frames/frame_temp.jpeg')
            
            im_frame = Image.open('Frames/frame_temp.jpeg')
            np_frame = np.array(im_frame)
            print(P,np_frame.shape)

            images.append(np_frame)

        
        return assignments, images

def convert_gif(imgs, filename, duration):
    """
    Convert a list of images to a gif
    Args:
        imgs:List.
            a list of images
        filename: str,
            the filename for the gif
        duration: int
            duration in ms between images in GIF
    Returns:
        None
    """

    stacked_images = []
    for img in imgs:
        stacked_images.append(Image.fromarray(np.asarray(img)))

    stacked_images[0].save(
        filename+".gif",
        optimize=False,
        save_all=True,
        append_images = stacked_images[1:],
        loop = 0,
        duration = duration
    )

clustering, images_temp = DBC(X, 3, .2).dbscan()
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
plt.scatter(X[:, 0], X[:, 1], color=colors[clustering].tolist(), s=10, alpha=0.8)
plt.title("Final Frame")

images = images + images_temp

plt.savefig('Frames/frame_last.jpeg')
im_frame = Image.open('Frames/frame_last.jpeg')
np_frame = np.array(im_frame)
images.append(np_frame)

convert_gif(images,"Animation",500)
