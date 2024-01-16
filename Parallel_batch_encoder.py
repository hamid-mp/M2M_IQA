import argparse
import concurrent.futures
import errno
import multiprocessing
import os
import subprocess
import sys
from PIL import Image

JPEG_EXEC_PATH="/home/user/hamid/vafa/compression/code/JPEG-XT/jpeg"

HEVC_ENCODER_EXEC_PATH="/home/user/hamid/vafa/compression/code/hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware/tags/HM-16.20+SCM-8.8/bin/TAppEncoderStatic"
HEVC_DECODER_EXEC_PATH="/home/user/hamid/vafa/compression/code/hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware/tags/HM-16.20+SCM-8.8/bin/TAppDecoderStatic"

VVC_ENCODER_EXEC_PATH="/home/user/hamid/vafa/compression/code/VVCSoftware_VTM/bin/EncoderAppStatic"
VVC_DECODER_EXEC_PATH="/home/user/hamid/vafa/compression/code/VVCSoftware_VTM/bin/DecoderAppStatic"

def sudo_run(command):
    #result = subprocess.run(command, capture_output= True, encoding='utf8')
    result = subprocess.run(['sudo'] + command, capture_output= True, encoding='utf8')
    print("Return Code:", result.returncode)
    print("Standard Output:", result.stdout)
    print("Standard Error:", result.stderr)
    return #result
    #return subprocess.run(['sudo'] + command, capture_output= True, encoding='utf8')

def img_dims(img_path):

    img = Image.open(img_path)
    w, h = img.size

    if w % 64 == 0  and h % 64 == 0:
        new_w, new_h = w, h

    else:
        new_w, new_h = round(w/64)*64, round(h/64)*64
        img = img.resize((new_w, new_h))
        img.save(img_path)

    return (new_w, new_h)



def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

def is_program_installed(prog):
    try:
        subprocess.run([str(prog), "-version"], check=True)
    except subprocess.CalledProcessError:
        print(f"{str(prog)} is not installed. Exiting...")
        sys.exit(1)

def check_software_installations(programs):
    for prog in programs:
        is_program_installed(prog)

check_software_installations(
    ["ffmpeg",
     "convert"])

def imagemagick_convert_image(input, output, after_decode=False, **kwargs):
    executable = "convert"
    if after_decode:
        cmd = [executable, str(input), str(output)]
    else:
        cmd = [executable, str(input), "-strip", str(output)]
    sudo_run(cmd)

def jpeg_encode(input, output, q, **kwargs):
    executable = JPEG_EXEC_PATH
    cmd = [executable, "-q", str(q), "-h", "-qt", "3",
           "-s", "1x1,2x2,2x2", str(input),
           str(output)]
    result = sudo_run(cmd)

def jpeg_decode(input, output, **kwargs):
    executable = JPEG_EXEC_PATH
    cmd = [executable, str(input), str(output)]

    sudo_run(cmd)

def ffmpeg_rgb_to_yuv(input, output, **kwargs):
    executable = "ffmpeg"
    cmd = [executable, "-hide_banner", "-i", str(input),
           "-pix_fmt", "yuv444p10le", "-vf",
           "scale=in_range=full:in_color_matrix=bt709:out_range=full:out_color_matrix=bt709",
           "-color_primaries", "bt709",
           "-color_trc", "bt709", "-colorspace", "bt709",
           "-y", str(output)]
    sudo_run(cmd)

def ffmpeg_yuv_to_rgb(input, output, w, h, **kwargs):
    executable = "ffmpeg"
    cmd = [executable, "-f", "rawvideo", "-vcodec",
           "rawvideo", "-s", f"{w}x{h}",
           "-r", "25", "-pix_fmt", "yuv444p10le",
           "-i", str(input), "-pix_fmt", "rgb24", 
           "-vf", "scale=in_range=full:in_color_matrix=bt709:out_range=full:out_color_matrix=bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-y", str(output)]
    sudo_run(cmd)

def hevc_encode(input, output, cfg, qp, w, h, **kwargs):
    executable = HEVC_ENCODER_EXEC_PATH

    cmd = [executable, "-c", str(cfg), "-i", str(input),
           "-wdt", str(w), "-hgt", str(h), "-b", output,
           "-f", "1", "-fr", "25", "-q", str(qp),
           "--FrameSkip=0", "--InputBitDepth=10",
           "--InputChromaFormat=444", "--ChromaFormatIDC=444", "--Level=6.2"]
    sudo_run(cmd)

def hevc_decode(input, output, **kwargs):
    print('-------------------HEVC DECODE-----------------')
    print(input, output)
    executable = HEVC_DECODER_EXEC_PATH
    cmd = [executable, "-d", "10", "-b", str(input),
           "-o", str(output)]
    sudo_run(cmd)

def vvc_encode(input, output, cfg, qp, w, h, **kwargs):
    executable = VVC_ENCODER_EXEC_PATH
    cmd = [executable, "-c", str(cfg), "-i", str(input),
           "-wdt", str(w), "-hgt", str(h), "-b", output,
           "-f", "1", "-fr", "25", "-q", str(qp),
           "--FrameSkip=0", "--InputBitDepth=10",
           "--InputChromaFormat=444", "--ChromaFormatIDC=444",
           "--TemporalSubsampleRatio=1", "--Level=6.2"]
    sudo_run(cmd)

def vvc_decode(input, output, **kwargs):
    executable = VVC_DECODER_EXEC_PATH
    cmd = [executable, "-d", "10", "-b", str(input),
           "-o", str(output)]
    sudo_run(cmd)

def convert_image_imagick(img_path, out_img_path, format, *args):
    if format == "pnm" or format == "png":
        imagemagick_convert_image(img_path, out_img_path, *args)

def convert_image_ffmpeg(img_path, out_img_path, format, *args):
    if format == "yuv":
        ffmpeg_rgb_to_yuv(img_path, out_img_path)
    elif format == "png":
        w, h = img_path.split('_wh_')[1].split('.')[0].split('_')
        ffmpeg_yuv_to_rgb(img_path, out_img_path, w, h)

def convert_all(in_path, out_path, codecs, filetypes):
    img_files = [f for f in os.listdir(in_path) if f.endswith(filetypes)]
    print(img_files)
    for codec in codecs:
        format = 'pnm' if codec == 'jpeg' else 'yuv'
        mkdir_p(os.path.join(out_path,codec,format))
        for img in img_files:
            #print(img, format)
            img_path = os.path.join(in_path, img)
            #image = Image.open(img_path)
            w, h = img_dims(img_path)
            #image.resize((w,h))
            #image.save(img_path)
            
            out_img_path = os.path.join(out_path,codec,format,
                                        os.path.splitext(img)[0]
                                        +f"_wh_{w}_{h}"+'.'+format)
            if format == 'pnm':
                convert_image_imagick(img_path, out_img_path, format)
            else:
                convert_image_ffmpeg(img_path, out_img_path, format)

def convert_back_all(in_path, out_path, format, filetypes):
    mkdir_p(out_path)
    img_files = [f for f in os.listdir(in_path) if f.endswith(filetypes)]
    for img in img_files:
        img_path = os.path.join(in_path, img)
        out_img_path = os.path.join(out_path,
                                    os.path.splitext(img)[0]+'.'+format)
        if format == 'pnm':
            convert_image_imagick(img_path, out_img_path, format, True)
        else:
            convert_image_ffmpeg(img_path, out_img_path, format)

def run_cmd(cmd, image_path):
    sudo_run(cmd + image_path)





def code_all_worker(args):
    in_path, out_path, format, out_format, coder_function, file, *extra_args = args

    input_file_path = os.path.join(in_path, file)
    w, h = input_file_path.split('_wh_')[1].split('.')[0].split('_')
    output_file_path = os.path.join(out_path, os.path.splitext(file)[0] + '.' + out_format)

    if coder_function == vvc_encode or coder_function == hevc_encode:
        coder_function(input_file_path, output_file_path, *extra_args, int(w), int(h))
    else:
        coder_function(input_file_path, output_file_path, *extra_args)

def code_all(in_path, out_path, format, out_format, coder_function, *args):
    mkdir_p(out_path)
    files = [f for f in os.listdir(in_path) if f.endswith(format)]

    # Create a list of arguments for each worker process
    worker_args_list = [(in_path, out_path, format, out_format, coder_function, file, *args) for file in files]

    # Number of parallel processes to run
    num_processes = multiprocessing.cpu_count()

    # Create a pool of processes
    with multiprocessing.Pool(num_processes) as pool:
        # Map the code_all_worker function to the list of worker arguments
        pool.map(code_all_worker, worker_args_list)

def jpeg_pipline(FLAGS):
    pnm_files_path = os.path.join(FLAGS.output,'jpeg','pnm')
    jpeg_bits_files_path = os.path.join(FLAGS.output,'jpeg','bits')
    pnm_decoded_files_path = os.path.join(FLAGS.output,'jpeg','pnm_decoded')
    png_files_path = os.path.join(FLAGS.output,'jpeg','png')
    
    code_all(pnm_files_path, jpeg_bits_files_path, 'pnm','bits', jpeg_encode, int(FLAGS.jpeg_quality))
    code_all(jpeg_bits_files_path, pnm_decoded_files_path,'bits','pnm', jpeg_decode)
    convert_back_all(pnm_decoded_files_path, png_files_path, 'png', ('.pnm',)) 

def vvc_pipeline(FLAGS):
    yuv_files_path = os.path.join(FLAGS.output,'vvc','yuv')
    vvc_bits_files_path = os.path.join(FLAGS.output,'vvc','bits')
    yuv_decoded_files_path = os.path.join(FLAGS.output,'vvc','yuv_decoded')
    png_files_path = os.path.join(FLAGS.output,'vvc','png')

    code_all(yuv_files_path, vvc_bits_files_path, 'yuv','bits', 
             vvc_encode, FLAGS.vvc_cfg, int(FLAGS.hevc_vvc_quality)
             )
    code_all(vvc_bits_files_path, yuv_decoded_files_path,'bits','yuv', vvc_decode)
    convert_back_all(yuv_decoded_files_path, png_files_path, 'png', ('.yuv',)) 

def hevc_pipeline(FLAGS):
    yuv_files_path = os.path.join(FLAGS.output,'hevc','yuv')
    hevc_bits_files_path = os.path.join(FLAGS.output,'hevc','bits')
    yuv_decoded_files_path = os.path.join(FLAGS.output,'hevc','yuv_decoded')
    png_files_path = os.path.join(FLAGS.output,'hevc','png')

    code_all(yuv_files_path, hevc_bits_files_path, 'yuv','bits', 
             hevc_encode, FLAGS.hevc_cfg, int(FLAGS.hevc_vvc_quality)
             )
    code_all(hevc_bits_files_path, yuv_decoded_files_path,'bits','yuv', hevc_decode)
    convert_back_all(yuv_decoded_files_path, png_files_path, 'png', ('.yuv',)) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',
                        help='input folder',
                        type=str,
                        default='./data/original')
    parser.add_argument('-o','--output',
                        help='output converted folder',
                        type=str,
                        default='./data/converted')
    parser.add_argument('-p','--preprocess',
                        type=bool,
                        help='convert original images to pnm and yuv',
                        default=True)
    parser.add_argument('-jpq','--jpeg_quality',
                        type=int,
                        help='jpeg encoder quality factor',
                        default=5)
    parser.add_argument('-hvq', '--hevc_vvc_quality',
                        type=float,
                        help='hevc, vvc encoder quality factor',
                        default=15)
    parser.add_argument('--vvc_cfg',
                        help='VVC encoder cfg file',
                        type=str,
                        default='/home/user/hamid/vafa/compression/code/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg')
    parser.add_argument('--hevc_cfg',
                        help='HEVC encoder cfg file',
                        type=str,
                        default='/home/user/hamid/vafa/compression/code/hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware/tags/HM-16.20+SCM-8.8/cfg/encoder_intra_main_scc_10.cfg')
    FLAGS = parser.parse_args()
    
    codecs = ['jpeg', 'hevc', 'vvc']
    if FLAGS.preprocess:
        convert_all(FLAGS.input, FLAGS.output, codecs,('.png', '.jpg', '.jpeg'))
    assert FLAGS.jpeg_quality > 1, 'You must use a numeric value for JPEG-Quality Factor'
    
    # TODO: a new separate process
    jpeg_pipline(FLAGS)
    vvc_pipeline(FLAGS)
    hevc_pipeline(FLAGS)

if __name__ == "__main__":
    main()
    print("PASS")



    