

ffmpeg -f image2 -i run%d.png vid.avi && ffmpeg -i vid.avi -filter:v "setpts=10.0*PTS" vidd.avi
ffmpeg -i vidd.avi -pix_fmt rgb24 out.gif && rm *.avi
