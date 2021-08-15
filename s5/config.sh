case "$USER" in
"tuananh")
  # Wav root (after unzipping)...
  export WAV_ROOT="./wav" 

  # Used by the recogniser for storing data/ exp/ mfcc/ etc
  export REC_ROOT="." 
  echo "Set wav and rec path done"
  ;;
*)
  echo "Please define WAV_ROOT and REC_ROOT for user $USER"
  ;;
esac

