addpath('/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_test_600/') 
HR=dir('/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_test_600/*.*');

addpath('/home/garciamorenc/tfm/Real-ESRGAN/results_x2_100k_5e-5lr_3batch/') 
Files=dir('/home/garciamorenc/tfm/Real-ESRGAN/results_x2_100k_5e-5lr_3batch/*.*');
% addpath('/home/garciamorenc/tfm/Real-ESRGAN/results_x2_original/') 
% Files=dir('/home/garciamorenc/tfm/Real-ESRGAN/results_x2_original/*.*');
% addpath('/home/garciamorenc/tfm/Real-ESRGAN/results_x4_x4_original/') 
% Files=dir('/home/garciamorenc/tfm/Real-ESRGAN/results_x4_x4_original/*.*');
% addpath('/home/garciamorenc/tfm/Real-ESRGAN/results_x4_x4_anime/') 
% Files=dir('/home/garciamorenc/tfm/Real-ESRGAN/results_x4_x4_anime/*.*');
% addpath('/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_test_600/') 
% Files=dir('/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_test_600/*.*');

list_psnr = [];
list_psnr_total = [];
list_psnr_arrugado = [];
list_psnr_roto = [];
list_ssim = [];
list_ssim_total = [];
list_ssim_arrugado = [];
list_ssim_roto = [];
list_brisque = [];
list_brisque_total = [];
list_brisque_arrugado = [];
list_brisque_roto = [];
list_niqe = [];
list_niqe_total = [];
list_niqe_arrugado = [];
list_niqe_roto = [];
list_piqe = [];
list_piqe_total = [];
list_piqe_arrugado = [];
list_piqe_roto = [];

for k=1:length(Files)
    FileNames=Files(k).name;
    HRNames=HR(k).name;
    if contains(FileNames, 'img')
%         disp(FileNames);
        A = imread(HRNames);
        B = imread(FileNames);
        [rowsA, columnsA, numberOfColorChannelsA] = size(A);
        [rowsB, columnsB, numberOfColorChannelsB] = size(B);
        if rowsA ~= rowsB        
            A = imresize(A,2);
        end
        
        peak_psnrs = psnr(A,B);
        if contains(FileNames, 'arrugado')
            list_psnr_arrugado = [list_psnr_arrugado, peak_psnrs];
        elseif contains(FileNames, 'roto')
            list_psnr_roto = [list_psnr_roto, peak_psnrs];
        else
            list_psnr = [list_psnr, peak_psnrs];
        end
        list_psnr_total = [list_psnr_total, peak_psnrs];
%         disp(peak_psnrs);
        
        ssimval = ssim(A,B);
        if contains(FileNames, 'arrugado')
            list_ssim_arrugado = [list_ssim_arrugado, ssimval];
        elseif contains(FileNames, 'roto')
            list_ssim_roto = [list_ssim_roto, ssimval];
        else
            list_ssim = [list_ssim, ssimval];
        end
        list_ssim_total = [list_ssim_total, ssimval];
        
%         disp(ssimval);
        
%         multissim3val = multissim(A,B);
%         disp(multissim3val);
        
        brisquescore = brisque(B);
        if contains(FileNames, 'arrugado')
            list_brisque_arrugado = [list_brisque_arrugado, brisquescore];
        elseif contains(FileNames, 'roto')
            list_brisque_roto = [list_brisque_roto, brisquescore];
        else
            list_brisque = [list_brisque, brisquescore];
        end
        list_brisque_total = [list_brisque_total, brisquescore];
%         disp(brisquescore);
        
        niqescore = niqe(B);
        if contains(FileNames, 'arrugado')
            list_niqe_arrugado = [list_niqe_arrugado, niqescore];
        elseif contains(FileNames, 'roto')
            list_niqe_roto = [list_niqe_roto, niqescore];
        else
            list_niqe = [list_niqe, niqescore];
        end
        list_niqe_total = [list_niqe_total, niqescore];
        
%         disp(niqescore);
        
        piqescore = piqe(B);
        if contains(FileNames, 'arrugado')
            list_piqe_arrugado = [list_piqe_arrugado, piqescore];
        elseif contains(FileNames, 'roto')
            list_piqe_roto = [list_piqe_roto, piqescore];
        else
            list_piqe = [list_piqe, piqescore];
        end
        list_piqe_total = [list_piqe_total, piqescore];
%         disp(piqescore);
    end    
end

M_psnr_total = mean(list_psnr_total);
M_psnr = mean(list_psnr);
M_psnr_arrugado = mean(list_psnr_arrugado);
M_psnr_roto = mean(list_psnr_roto);
M_ssim_total = mean(list_ssim_total);
M_ssim = mean(list_ssim);
M_ssim_arrugado = mean(list_ssim_arrugado);
M_ssim_roto = mean(list_ssim_roto);
M_brisque_total = mean(list_brisque_total);
M_brisque = mean(list_brisque);
M_brisque_arrugado = mean(list_brisque_arrugado);
M_brisque_roto = mean(list_brisque_roto);
M_niqe_total = mean(list_niqe_total);
M_niqe = mean(list_niqe);
M_niqe_arrugado = mean(list_niqe_arrugado);
M_niqe_roto = mean(list_niqe_roto);
M_piqe_total = mean(list_piqe_total);
M_piqe = mean(list_piqe);
M_piqe_arrugado = mean(list_piqe_arrugado);
M_piqe_roto = mean(list_piqe_roto);
    
txt_psnr = sprintf('psnr mean: \n sintetico %.4f, \n arrugado %.4f, \n roto %.4f, \n total %.4f', M_psnr, M_psnr_arrugado, M_psnr_roto, M_psnr_total);
disp(txt_psnr);
txt_ssim = sprintf('ssim mean: \n sintetico %.4f, \n arrugado %.4f, \n roto %.4f, \n total %.4f', M_ssim, M_ssim_arrugado, M_ssim_roto, M_ssim_total);
disp(txt_ssim);
txt_brisque = sprintf('brisque mean: \n sintetico %.4f, \n arrugado %.4f, \n roto %.4f, \n total %.4f', M_brisque, M_brisque_arrugado, M_brisque_roto, M_brisque_total);
disp(txt_brisque);
txt_niqe = sprintf('niqe mean: \n sintetico %.4f, \n arrugado %.4f, \n roto %.4f, \n total %.4f', M_niqe, M_niqe_arrugado, M_niqe_roto, M_niqe_total);
disp(txt_niqe);
txt_piqe = sprintf('piqe mean: \n sintetico %.4f, \n arrugado %.4f, \n roto %.4f, \n total %.4f', M_piqe, M_piqe_arrugado, M_piqe_roto, M_piqe_total);
disp(txt_piqe);