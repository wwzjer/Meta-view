import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import cv2
from scipy.misc import imresize


def visualize(state, visited_idxes,action_probs_all,args,sample_idx=None,task_id=None):
    num_id = state.views.shape[0]
    azi = state.views.shape[1]
    ele = state.views.shape[2]
    for id in range(num_id):
        view_grid = np.zeros((azi*84,ele*84))
        for i in range(azi):
            for j in range(ele):
                visited_idx = [list(i) for i in list(np.array(visited_idxes)[:,id,:])]
                if [i,j] not in visited_idx:
                    view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = 0.5*state.views[id,i,j,0,:,:]          # unvisited views are darken
                else:
                    view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = state.views[id,i,j,0,:,:]
                for t in range(len(visited_idx)):
                    m,n = visited_idx[t]
                    prob_grid = imresize(action_probs_all[0][0,:].cpu().detach().numpy().reshape(5,5),(25,25), interp='nearest')/255
                    prob_grid[0:25:5, :] = 1
                    prob_grid[:, 0:25:5] = 1
                    view_grid[m*84:m*84+25,n*84:n*84+25] = prob_grid
                    if t == 0:
                        view_grid[(m+1)*84-25:(m+1)*84,(n+1)*84-25:(n+1)*84] = 1-plt.imread('1.png')
                    elif t == 1:
                        view_grid[(m+1)*84-25:(m+1)*84,(n+1)*84-25:(n+1)*84] = 1-plt.imread('2.png')
                    elif t == 2:
                        view_grid[(m+1)*84-25:(m+1)*84,(n+1)*84-25:(n+1)*84] = 1-plt.imread('3.png')
        if not os.path.exists('visualize/'+str(sample_idx)+'/'):
            os.mkdir('visualize/'+str(sample_idx)+'/')
        if not os.path.exists('visualize/'+str(sample_idx)+'/'+str(task_id)+'/'):
            os.mkdir('visualize/'+str(sample_idx)+'/'+str(task_id)+'/')       
        if num_id == args.num_classes_per_set:
            plt.imsave('visualize/'+str(sample_idx)+'/'+str(task_id)+'/'+'/support_'+str(id),view_grid,cmap='gray',vmin=view_grid.min(),vmax=view_grid.max())
        else:
            plt.imsave('visualize/'+str(sample_idx)+'/'+str(task_id)+'/'+'/target_'+str(id),view_grid,cmap='gray',vmin=view_grid.min(),vmax=view_grid.max())


# action_probs_all[0][0,:].cpu().detach().numpy().reshape(5,5)
# imresize(action_probs_all[0][0,:].cpu().detach().numpy().reshape(5,5),(25,25))

# def visualize(x,support_visited_idxes,target_visited_idxes,args):

#     num_id = x.views.shape[0]
#     azi = x.views.shape[1]
#     ele = x.views.shape[2]
#     dic = {}
#     for id in range(num_id):
#         # view_grid = np.zeros((5*84,6*84))
#         view_grid = np.zeros((30*84,84))
#         num = 0
#         for i in range(azi):
#             for j in range(ele):
#                 view_grid[num*84:(num+1)*84,:] = x.views[id,i,j,0,:,:]
#                 dic[num] = [i,j]
#                 num += 1
#                 # view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = x.views[id,i,j,0,:,:]
#         view_grid = np.tile(view_grid,1+15)
    
#         for t in range(len(support_visited_idxes)):
#             m,n = support_visited_idxes[t][id]
#             num = [k for k,v in dic.items() if v==[m,n]]
#             view_grid[num[0]*84:(num[0]+1)*84,0:84] += 0.5
#             for target in range(15):
#                 k,l = target_visited_idxes[t][target]
#                 num = [a for a,b in dic.items() if b==[k,l]]
#                 view_grid[num[0]*84:(num[0]+1)*84,(target+1)*84:(target+2)*84] += 0.5

#         plt.imsave('visualize/'+str(id),view_grid)
#         # if num_id == args.num_classes_per_set:
#         #     plt.imsave('visualize/support_'+str(id),view_grid)
#         # else:
#         #     plt.imsave('visualize/target_'+str(id),view_grid)


# # def visualize(x,visited_idxes,opts,writer,epoch):   
# #     # ================================================================
# #     #                      Visualization part
# #     # ================================================================
# #     # Note: visited_idxes_val = B-D list of T-D list of batch_size-D list of 2-D list
# #     vis_samples = 3
# #     # Create empty canvas
# #     vis_tensors = torch.zeros((vis_samples * opts.batch_size, opts.T, opts.N, opts.M, 30, 30, 1))
# #     # fill in borders (only one vertical and horizontal border)
# #     vis_tensors[:, :, :, :, 0, :, :] = 1
# #     vis_tensors[:, :, :, :, :, 0, :] = 1
# #     for b1 in range(vis_samples):
# #         batch_size_curr = len(visited_idxes[b1][0])
# #         for b2 in range(batch_size_curr):
# #             for t in range(opts.T):
# #                 vis_tensors[b1*opts.batch_size + b2, t, visited_idxes[b1][t][b2][0], visited_idxes[b1][t][b2][1], :, :, :] = 0.5
# #     # Convert to B x T x N*10 x M*10 x 1
# #     vis_tensors = torch.Tensor.permute(vis_tensors, 0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, opts.T, opts.N*30, opts.M*30, 1)
# #     # Convert to B x T x 1 X 10N x 10M
# #     vis_tensors = torch.Tensor.permute(vis_tensors, 0, 1, 4, 2, 3).contiguous()
# #     vis_tensors_size = vis_tensors.size(0)
# #     # List of T x 1 x 10N x 10M tensors
# #     vis_tensors = [torchvision.utils.make_grid(vis_tensors[b], opts.T, padding=4, pad_value=1.0) \
# #                                                                         for b in range(vis_tensors_size)]
# #     for b in range(vis_tensors_size):
# #         writer.add_image('Validation Panorama #%d'%(b), vis_tensors[b], epoch+1)


# import matplotlib.pyplot as plt
# for k in range(5):
#     view_grid = np.zeros((5*84,6*84))
#     for i in range(5):
#         for j in range(6):
#             if [i,j] == list(pro[k]):
#                 view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = x.views[k,i,j]
#             else:
#                 view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = 0.2*x.views[k,i,j]
#     plt.imsave('visualize/support_'+str(k)+'1',view_grid,cmap='gray',vmin=view_grid.min(),vmax=view_grid.max())

# import matplotlib.pyplot as plt
# for k in range(5):
#     view_grid = np.zeros((5*84,6*84))
#     for i in range(5):
#         for j in range(6):
#             if [i,j] == list(pro[15*k]):
#                 view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = x.views[15*k,i,j]
#             else:
#                 view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = 0.2*x.views[15*k,i,j]
#     plt.imsave('visualize/target_'+str(k)+'3',view_grid,cmap='gray',vmin=view_grid.min(),vmax=view_grid.max())

# # for k in range(5):
# #     k = 0
# #     view_grid = np.zeros((5*84,6*84))
# #     for i in range(5):
# #         for j in range(6):
# #             if [i,j] == list(pro[k*15]):
# #                 view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = x.views[k*15,i,j]
# #             else:
# #                 view_grid[i*84:(i+1)*84,j*84:(j+1)*84] = 0.*x.views[k*15,i,j]
# #     d = [list(self.args.act_to_delta[i]) for i in range(self.args.A)]
# #     idx = [[np.clip(pro[k*15][0].cpu().numpy() + d[i][0], 0, 5-1),(pro[k*15][1].cpu().numpy() + d[i][1])%6] for i in self.args.act_to_delta.keys()]
# #     for num in range(len(idx)):
# #         idxx = idx[num]
# #         m, n = np.int(idxx[0]), np.int(idxx[1])
# #         if np.array([m]) == pro[k*15][0].cpu().numpy().astype(np.int) and np.array([n]) == pro[k*15][1].cpu().numpy().astype(np.int):
# #             print(m,n,True)
# #             view_grid[m*84:(m+1)*84,n*84:(n+1)*84] = x.views[k*15,m,n]
# #         else:
# #             print(m,n,False)
# #             view_grid[m*84:(m+1)*84,n*84:(n+1)*84] = x.views[k*15,m,n] * action_probs[k*15][num].cpu().detach().numpy()    

# #     plt.imsave('visualize/target_'+str(k)+'1',view_grid,cmap='gray',vmin=view_grid.min(),vmax=view_grid.max())

    
