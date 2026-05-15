import torch 
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from utils.utils import *
from utils.models import *
import torch.optim as optim 
from tqdm import tqdm 
from torchvision.utils import save_image

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default= 'C:/Users/gokup/OneDrive/Desktop/NST/NST_code/content_data',
                        help='Location of content data')
    
    parser.add_argument('--style_dir',type=str,default="C:/Users/gokup/OneDrive/Desktop/NST/NST_code/style_data",
                        help ='Location of style dataset')
    
    parser.add_argument('--vgg',type=str,default="C:/Users/gokup/OneDrive/Desktop/NST/NST_code/vgg_normalised.pth",
                        help='Location of pre-trained Vgg')
    
    parser.add_argument('--experiment',type=str, default="experiment_1",
                        help='Name of experiment')    #palce where our model save
    
    parser.add_argument('--final_size',type=int, default=256,
                        help='size of final image')
    
    parser.add_argument('--content_size',type=int, default=512,
                        help='size of content image')
    
    parser.add_argument('--style_size',type=int, default=512,
                        help='size of style image')
    
    parser.add_argument('--crop',action='store_true', default=True,
                        help='crop image')
    
    parser.add_argument('--batch_size',type=int, default=4,
                        help='Number of batch')
    
    parser.add_argument('--lr',type=float, default=1e-4,
                        help='Learning rate')
    
    parser.add_argument('--lr_decay',type=float, default=5e-5,
                        help='Learning rate decay')
    
    parser.add_argument('--epochs',type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--start_epochs',type=int, default=0,
                        help='Number of epochs')
    
    parser.add_argument('--content_weight', type=float , default=1.0,
                        help='content weight')
    
    parser.add_argument('--style_weight', type=float , default=5,
                        help='style weight')
    
    parser.add_argument('--log_interval', type=int , default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int , default=2,
                        help='Save interval')
    
    parser.add_argument('--resume',action='store_true', default=False,
                        help='Resume Training')
    
    parser.add_argument('--decoder_path',type=str,default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path',type=str,default=None,
                        help='Path to optimizer checkpoint')
    

    return parser.parse_args()


def main():
    args = parse_argument()
    
    #device setup 
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    save_dir = Path('experiment')/ args.experiment  #path
    save_dir.mkdir(exist_ok=True,parents=True)      #create_folder 

    #save argument values inside save_dir folder
    with open(save_dir / 'args.txt' , 'w') as f:
        for key , value in vars(args).items():
            f.write(f'{key}:{value}\n')


    #datset
    content_tranformation = get_transforms(args.content_size,args.crop,args.final_size)
    style_tranformation = get_transforms(args.style_size,args.crop,args.final_size)
    

    content_dataset = ImageFolderDataset(args.content_dir,content_tranformation)
    style_dataset = ImageFolderDataset(args.style_dir,style_tranformation)

    #dataloader
    content_dataloader = DataLoader(content_dataset,
                                batch_size= args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    style_dataloader = DataLoader(style_dataset,
                              batch_size= args.batch_size,
                              shuffle=True ,
                              pin_memory=True,
                              drop_last=True)


    print('Number of batches in content dataset',len(content_dataloader))
    print('Number of batches in style dataset',len(style_dataloader))

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda= lambda epoch : 1.0/(1.0 + args.lr_decay * epoch)
    )

    #checkpoint loader
    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path,map_location=torch.device('cpu')))
        optimizer.load_state_dict(torch.load(args.optimizer_path,map_location=torch.device('cpu')))


    # Train our model 
    print("Training......")

    mse_loss = nn.MSELoss()

    encoder.eval()

    running_loss = None
    running_content_loss = None 
    running_style_loss = None 

    for epoch in range(args.start_epochs,args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total= min(len(content_dataloader), len(style_dataloader)))
        
        running_loss = 0
        running_content_loss = 0
        running_style_loss = 0 
        
        for content_batch , style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_features = encoder(content_batch)
            s_features = encoder(style_batch)

            t = AdaIN(c_features[-1],s_features[-1])

            gen_img = decoder(t)

            gen_img_features = encoder(gen_img)

            content_loss = mse_loss(gen_img_features[-1],t) * args.content_weight

            style_loss = 0
            for s_f , g_f in zip(s_features,gen_img_features):
                g_mean , g_std = cal_mean_std(g_f)
                s_mean , s_std = cal_mean_std(s_f)
                style_loss += mse_loss(g_mean, s_mean) + mse_loss(g_std,s_std)
            
            style_loss = style_loss * args.style_weight

            loss = content_loss + style_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss: {loss.item():4f}, content loss: {content_loss.item():4f}, style loss: {style_loss.item():4f}')

            running_loss+= loss.item()
            running_content_loss += content_loss.item()
            running_style_loss += style_loss.item()

        scheduler.step()

        running_loss /= len(content_dataloader)
        running_content_loss /= len(content_dataloader)
        running_style_loss /= len(content_dataloader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1} : Loss= {running_loss:4f} , content_loss = {running_content_loss} , style_loss = {running_style_loss}')

        
        #save model 
        if (epoch+1) % args.save_interval ==0:
            torch.save(decoder.state_dict(),save_dir/f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(),save_dir/f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, gen_img], dim=0)
                save_image(output , save_dir/f'output_{epoch+1}.png',nrow= args.batch_size)




if __name__ == '__main__':
    main()