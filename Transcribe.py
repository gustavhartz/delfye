import whisper
import subprocess
import argparse
from pydub import AudioSegment
import os
from huggingface_hub import login
from pyannote.audio import Pipeline
import whisper
import torch
import json
import re
import time
from tqdm import tqdm
import locale
from datetime import timedelta
locale.getpreferredencoding = lambda: "UTF-8"
SPACER_LENGTH_MILISECONDS = 2000
INPUT_FILE_NAME = "input.wav"
INPUT_PREP_FILE_NAME = "input_prep.wav"


def convert_audio_file_to_format(args) -> None:
    subprocess.Popen(f"ffmpeg -i {os.path.join(args.file_path,args.file_name)} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y {os.path.join(args.output_dir, INPUT_FILE_NAME)}",
                     shell=True, stdout=subprocess.PIPE).stdout.read()


def prepend_spacer(args) -> None:
    """
    Prepend a silence to the beginning of the audio file
    """
    spacer = AudioSegment.silent(duration=SPACER_LENGTH_MILISECONDS)
    audio = AudioSegment.from_wav(
        os.path.join(args.output_dir, INPUT_FILE_NAME))
    audio = spacer.append(audio, crossfade=0)
    audio.export(os.path.join(args.output_dir,
                 INPUT_PREP_FILE_NAME), format='wav')


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)(
        (int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000
    )
    return s


def group_by_speaker(dzs: str):

    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
            groups.append(g)
            g = []

        g.append(d)
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=d)[1]
        end = millisec(end)
        if lastend > end:  # segment engulfed by a previous segment
            groups.append(g)
            g = []
        else:
            lastend = end
    if g:
        groups.append(g)
    return groups


def save_audio_segments(audio: AudioSegment, groups: list, args) -> None:
    for gidx, g in enumerate(groups):
        start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
        start = millisec(start)  # - spacermilli
        end = millisec(end)  # - spacermilli
        audio[start:end].export(os.path.join(
            args.output_dir, f"{gidx}.wav"), format="wav")


def transcribe_grouped_audio(model: whisper.Whisper, groups, args) -> None:
    """
    Transcribe audio segments
    """
    combined_file = {}
    for i in tqdm(range(len(groups)), desc="transcribing segments"):
        base_path = os.path.join(args.output_dir, str(i))
        audiof = base_path+".wav"
        result = model.transcribe(
            audio=audiof, word_timestamps=True, verbose=args.verbose, language=args.language)
        combined_file["segment_"+str(i)] = result
        with open(base_path+".json", "w") as outfile:
            json.dump(result, outfile, indent=4)
    json.dump(combined_file, open(os.path.join(
        args.output_dir, "combined_file.json"), "w"), indent=4)


def assert_env_defined(env_var: str) -> None:
    """
    Assert that an environment variable is defined
    """
    if not os.getenv(env_var):
        raise Exception(f"Environment variable {env_var} is not defined")


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(args):
    """
    Transcribe the audio file and save the segments for each speaker
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")
    if args.device != DEVICE:
        print(
            f"args.device: {args.device} unavailable. Defaulting to {DEVICE}")
    ensure_directory_exists(args.output_dir)
    convert_audio_file_to_format(args)
    assert_env_defined("HUGGINGFACE_API_KEY")
    prepend_spacer(args)
    login(token=os.getenv("HUGGINGFACE_API_KEY"))
    pipeline = Pipeline.from_pretrained(
        'pyannote/speaker-diarization', use_auth_token=True)
    if not pipeline:
        raise Exception("Pipeline not found or inaccisable")
    pipeline.to(args.device)
    print("Pipeline loaded - annotate")
    t1 = time.time()
    dz = pipeline(os.path.join(args.output_dir, INPUT_PREP_FILE_NAME))
    # print time used
    print(f"Time used on annotation: {time.time()-t1}")
    with open(os.path.join(args.output_dir, "diarization.json"), "w") as outfile:
        outfile.write(str(dz))

    groups = group_by_speaker(str(dz).split("\n"))
    # Save the audio segments for each speaker
    audio = AudioSegment.from_wav(os.path.join(
        args.output_dir, INPUT_PREP_FILE_NAME))
    save_audio_segments(audio, groups, args)
    # MEmory cleanup
    del pipeline, audio, dz
    # import whisper and run on audio files
    print("Loading whisper model")
    t1 = time.time()
    model = whisper.load_model(args.whisper_model, device=DEVICE)
    print(f"Time used on loading whisper: {time.time()-t1}")
    t1 = time.time()
    print("Transcribing audio")
    transcribe_grouped_audio(model, groups, args)
    print(f"Time used on loading whisper: {time.time()-t1}")
    print("Done")
    if args.save_html:
        print("Saving HTML")
        convert_output_to_html_and_txt(groups, args)


def convert_output_to_html_and_txt(groups, args) -> None:
    """
    Convert the output to HTML
    """
    audio_title = args.file_name
    preS = '\n<!DOCTYPE html>\n<html lang="en">\n\n<head>\n\t<meta charset="UTF-8">\n\t<meta name="viewport" content="whtmlidth=device-width, initial-scale=1.0">\n\t<meta http-equiv="X-UA-Compatible" content="ie=edge">\n\t<title>' + \
        audio_title + \
        '</title>\n\t<style>\n\t\tbody {\n\t\t\tfont-family: sans-serif;\n\t\t\tfont-size: 14px;\n\t\t\tcolor: #111;\n\t\t\tpadding: 0 0 1em 0;\n\t\t\tbackground-color: #efe7dd;\n\t\t}\n\n\t\ttable {\n\t\t\tborder-spacing: 10px;\n\t\t}\n\n\t\tth {\n\t\t\ttext-align: left;\n\t\t}\n\n\t\t.lt {\n\t\t\tcolor: inherit;\n\t\t\ttext-decoration: inherit;\n\t\t}\n\n\t\t.l {\n\t\t\tcolor: #050;\n\t\t}\n\n\t\t.s {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.c {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.e {\n\t\t\t/*background-color: white; Changing background color */\n\t\t\tborder-radius: 10px;\n\t\t\t/* Making border radius */\n\t\t\twidth: 50%;\n\t\t\t/* Making auto-sizable width */\n\t\t\tpadding: 0 0 0 0;\n\t\t\t/* Making space around letters */\n\t\t\tfont-size: 14px;\n\t\t\t/* Changing font size */\n\t\t\tmargin-bottom: 0;\n\t\t}\n\n\t\t.t {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t#player-div {\n\t\t\tposition: sticky;\n\t\t\ttop: 20px;\n\t\t\tfloat: right;\n\t\t\twidth: 40%\n\t\t}\n\n\t\t#player {\n\t\t\taspect-ratio: 16 / 9;\n\t\t\twidth: 100%;\n\t\t\theight: auto;\n\t\t}\n\n\t\ta {\n\t\t\tdisplay: inline;\n\t\t}\n\t</style>'
    preS += '\n\t<script>\n\twindow.onload = function () {\n\t\t\tvar player = document.getElementById("audio_player");\n\t\t\tvar player;\n\t\t\tvar lastword = null;\n\n\t\t\t// So we can compare against new updates.\n\t\t\tvar lastTimeUpdate = "-1";\n\n\t\t\tsetInterval(function () {\n\t\t\t\t// currentTime is checked very frequently (1 millisecond),\n\t\t\t\t// but we only care about whole second changes.\n\t\t\t\tvar ts = (player.currentTime).toFixed(1).toString();\n\t\t\t\tts = (Math.round((player.currentTime) * 5) / 5).toFixed(1);\n\t\t\t\tts = ts.toString();\n\t\t\t\tconsole.log(ts);\n\t\t\t\tif (ts !== lastTimeUpdate) {\n\t\t\t\t\tlastTimeUpdate = ts;\n\n\t\t\t\t\t// Its now up to you to format the time.\n\t\t\t\t\tword = document.getElementById(ts)\n\t\t\t\t\tif (word) {\n\t\t\t\t\t\tif (lastword) {\n\t\t\t\t\t\t\tlastword.style.fontWeight = "normal";\n\t\t\t\t\t\t}\n\t\t\t\t\t\tlastword = word;\n\t\t\t\t\t\t//word.style.textDecoration = "underline";\n\t\t\t\t\t\tword.style.fontWeight = "bold";\n\n\t\t\t\t\t\tlet toggle = document.getElementById("autoscroll");\n\t\t\t\t\t\tif (toggle.checked) {\n\t\t\t\t\t\t\tlet position = word.offsetTop - 20;\n\t\t\t\t\t\t\twindow.scrollTo({\n\t\t\t\t\t\t\t\ttop: position,\n\t\t\t\t\t\t\t\tbehavior: "smooth"\n\t\t\t\t\t\t\t});\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}, 0.1);\n\t\t}\n\n\t\tfunction jumptoTime(timepoint, id) {\n\t\t\tvar player = document.getElementById("audio_player");\n\t\t\thistory.pushState(null, null, "#" + id);\n\t\t\tplayer.pause();\n\t\t\tplayer.currentTime = timepoint;\n\t\t\tplayer.play();\n\t\t}\n\t\t</script>\n\t</head>'
    preS += '\n\n<body>\n\t<h2>' + audio_title + '</h2>\n\t<i>Click on a part of the transcription, to jump to its portion of audio, and get an anchor to it in the address\n\t\tbar<br><br></i>\n\t<div id="player-div">\n\t\t<div id="player">\n\t\t\t<audio controls="controls" id="audio_player">\n\t\t\t\t<source src="input.wav" />\n\t\t\t</audio>\n\t\t</div>\n\t\t<div><label for="autoscroll">auto-scroll: </label>\n\t\t\t<input type="checkbox" id="autoscroll" checked>\n\t\t</div>\n\t</div>\n'

    postS = '\t</body>\n</html>'

    def timeStr(t):
        return '{0:02d}:{1:02d}:{2:06.2f}'.format(round(t // 3600),
                                                  round(t % 3600 // 60),
                                                  t % 60)

    html = list(preS)
    txt = list("")
    gidx = -1
    for g in groups:
        shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
    # the start time in the original video
        shift = millisec(shift) - SPACER_LENGTH_MILISECONDS
        shift = max(shift, 0)

        gidx += 1

        captions = json.load(
            open(os.path.join(args.output_dir, str(gidx) + '.json')))['segments']

        if captions:
            speaker = g[0].split()[-1]
            boxclr = "white"
            spkrclr = "orange"

            html.append(
                f'<div class="e" style="background-color: {boxclr}">\n')
            html.append(
                '<p  style="margin:0;padding: 5px 10px 10px 10px;word-wrap:normal;white-space:normal;">\n')
            html.append(
                f'<span style="color:{spkrclr};font-weight: bold;">{speaker}</span><br>\n\t\t\t\t')

            for c in captions:
                start = shift + c['start'] * 1000.0
                start = start / 1000.0  # time resolution ot youtube is Second.
                end = (shift + c['end'] * 1000.0) / 1000.0
                txt.append(
                    f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n')

                for i, w in enumerate(c['words']):
                    if w == "":
                        continue
                    start = (shift + w['start']*1000.0) / 1000.0
                    # end = (shift + w['end']) / 1000.0   #time resolution ot youtube is Second.
                    html.append(
                        f'<a href="#{timeStr(start)}" id="{"{:.1f}".format(round(start*5)/5)}" class="lt" onclick="jumptoTime({int(start)}, this.id)">{w["word"]}</a><!--\n\t\t\t\t-->')
                # html.append('\n')
            html.append('</p>\n')
            html.append(f'</div>\n')
    html.append(postS)

    with open(os.path.join(args.output_dir, "capspeaker.txt"), "w", encoding='utf-8') as file:
        s = "".join(txt)
        file.write(s)
        print('captions saved to capspeaker.txt:')

    with open(os.path.join(args.output_dir, "capspeaker.html"), "w", encoding='utf-8') as file:
        s = "".join(html)
        file.write(s)
        print('captions saved to capspeaker.html:')


"""
Main function call. Argpase to accept the file name and a bool to annotate speakers
"""
if __name__ == "__main__":
    # argpase
    parser = argparse.ArgumentParser(description='Process hyper-parameters')

    parser.add_argument("--file_name", help="File name to transcribe",
                        type=str)

    parser.add_argument("--file_path", help="File name to transcribe",
                        type=str, default="./", const=True, nargs='?')

    parser.add_argument("--device", help="Device to use",
                        type=str, default="cpu", const=True, nargs='?')

    parser.add_argument("--annotate", help="Annotate speakers",
                        type=bool, default=True, const=True, nargs='?')

    parser.add_argument("--language", help="Language to use",
                        type=str, default=None, const=True, nargs='?')

    parser.add_argument("--whisper_model", help="Model to use",
                        type=str, default="base", const=True, nargs='?')

    parser.add_argument("--verbose", help="Logging of whisper model",
                        type=bool, default=False, const=True, nargs='?')

    parser.add_argument("--save_html", help="Save an HTML file showing the assumed transcription",
                        type=bool, default=True, const=True, nargs='?')

    parser.add_argument("--cleanup", help="Delete various intermediate files",
                        type=bool, default=False, const=True, nargs='?')
    # outdir
    parser.add_argument("--output_dir", help="Output directory",
                        type=str, default="./out", const=True, nargs='?')

    # parse
    args = parser.parse_args()
    main(args)
