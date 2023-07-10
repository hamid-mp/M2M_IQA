import argparse
import concurrent.futures
import errno
import os
import subprocess
import sys
from pathlib import Path
JPEG_EXEC_PATH="/home/user/hamid/vafa/compression/code/JPEG-XT/jpeg"
JPEG_Q=0.1

HEVC_ENCODER_EXEC_PATH="path/to/hevc_encoder"
HEVC_DECODER_EXEC_PATH="path/to/hevc_decoder"

VVC_ENCODER_EXEC_PATH="path/to/vvc_encoder"
VVC_DECODER_EXEC_PATH="path/to/vvc_decoder"




def sudo_run(command):
    return subprocess.run(['sudo'] + command, capture_output= True, encoding='utf8')


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
    #print(result.returncode, result.stdout, result.stderr)


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
           "color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-y", str(output)]
    sudo_run(cmd)

def hevc_enconde(input, output, cfg, w, h, qp, **kwargs):
    executable = HEVC_ENCODER_EXEC_PATH
    cmd = [executable, "-c", str(cfg), "-i", str(input),
           "-wdt", str(w), "-hgt", str(h), "-b", output,
           "-f", "1", "-fr", "25", "-q", str(qp),
           "--FrameSkip=0", "--InputBitDepth=10",
           "--InputChromaFormat=444", "--ChromaFormatIDC=444", "--Level=6.2"]
    sudo_run(cmd)

def hevc_decode(input, output, **kwargs):
    executable = HEVC_DECODER_EXEC_PATH
    cmd = [executable, "-d", "10", "-b", str(input),
           "-r", str(output)]
    sudo_run(cmd)

def vvc_encode(input, output, cfg, w, h, qp, **kwargs):
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
           "-r", str(output)]
    sudo_run(cmd)

def convert_image(img_path, out_img_path, format, *args):
    if format == "pnm" or format == "png":


        imagemagick_convert_image(img_path, out_img_path, *args)
    elif format == "yuv":
        ffmpeg_rgb_to_yuv(img_path, out_img_path)

def convert_all(in_path, out_path, codecs, filetypes):
    #print(in_path)
    img_files = [f for f in os.listdir(in_path) if f.endswith(filetypes)]


    for codec in codecs:
        format = 'pnm' if codec == 'jpeg' else 'yuv'

        mkdir_p(os.path.join(out_path,codec,format))
        for img in img_files:
            img_path = os.path.join(in_path, img)
            out_img_path = os.path.join(out_path,codec,format,
                                        os.path.splitext(img)[0]+'.'+format)
            convert_image(img_path, out_img_path, format)

def convert_back_all(in_path, out_path, format, filetypes):
    mkdir_p(out_path)
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    img_files = [f for f in os.listdir(in_path) if f.endswith(filetypes)]
    for img in img_files:
        img_path = os.path.join(in_path, img)
        out_img_path = os.path.join(out_path,
                                    os.path.splitext(img)[0]+'.'+format)
        convert_image(img_path, out_img_path, format, True)

def run_cmd(cmd, image_path):
    sudo_run(cmd + image_path)

def code_all(in_path, out_path, format, out_format, coder_function, *args):
    files = [f for f in os.listdir(in_path) if f.endswith(format)]

    mkdir_p(out_path)

    for file in files:
        input_file_path = os.path.join(in_path, file)
        output_file_path = os.path.join(out_path,
                                    os.path.splitext(file)[0]+'.'+ out_format)
        
        coder_function(input_file_path, output_file_path, *args)






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
    
    parser.add_argument('-jpq', '--jpeg_quality', type=int, default=5)
    parser.add_argument('-hvq', '--hevc_vvc_quality', type=float, default=10)

    FLAGS = parser.parse_args()
    
    codecs = ['jpeg', 'hevc', 'vvc']
    if FLAGS.preprocess:
        convert_all(FLAGS.input, FLAGS.output, codecs,('.png', '.jpg', '.jpeg'))

    assert FLAGS.jpeg_quality > 1, 'You must use a numeric value for JPEG-Quality Factor'

    # TODO: a new separate process
    pnm_files_path = os.path.join(FLAGS.output,'jpeg','pnm')
    jpeg_bits_files_path = os.path.join(FLAGS.output,'jpeg','bits')
    pnm_decoded_files_path = os.path.join(FLAGS.output,'jpeg','pnm_decoded')
    png_files_path = os.path.join(FLAGS.output,'jpeg','png')
    code_all(pnm_files_path, jpeg_bits_files_path, 'pnm','bits', jpeg_encode, int(FLAGS.jpeg_quality))
    code_all(jpeg_bits_files_path, pnm_decoded_files_path,'bits','pnm', jpeg_decode)
    convert_back_all(pnm_decoded_files_path, png_files_path, 'png', ('.pnm',)) 

    # ...


if __name__ == "__main__":
    main()
