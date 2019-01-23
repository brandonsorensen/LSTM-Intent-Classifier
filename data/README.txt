Dialog act classification: an utterance classification task.

Files:

- utterances.train  
- utterances.valid

  Format:
  utterance_ID   dialog_act   utterance_t-3 ; utterance_t-2 ; utterance_t-1 ; utterance_t 


- utterances.test

  Format:
  utterance_ID   utterance_t-3 ; utterance_t-2 ; utterance_t-1 ; utterance_t 



Notes:
* the dialog_act corresponds to the last utterance of each line (utterance_t)
* utterance_t-3, utterance_t-2 and utterance_t-1  are provided if you want to use the context 
* utterance_t-1 : previous utterance
* utterance_t-1 : second previous utterance
* utterance_t-1 : third previous utterance


Dialog act set

%   :  Indecipherable
%-- :  Abandoned
2   :  Collaborative-Completion
aa  :  Accept, Yes Answers
aap :  Partial Accept
ar  :  Reject, No Answers
b   :  Backchannel
ba  :  Appreciation
bc  :  Misspeak Correction
bd  :  Downplayer
bh  :  Rhetorical-Question Backchannel
bk  :  Acknowledgment
br  :  Signal Non understanding
bs  :  Reformulation
cc  :  Commit (self-inclusive)
co  :  Command
d   :  Declarative-Question
fa  :  Apology
ft  :  Thanks
g   :  Tag Question
h   :  Hold
no  :  No knowledge answers
qh  :  Rhetorical Question
qo  :  Open-Ended Question
qrr :  Or Clause After Y/N Question
qw  :  Wh- Question
qy  :  Y/N Question
s   :  Statement
t1  :  Self-Talk
t3  :  3rd-Party Talk
x   :  Nonspeech


Text file format for submission (matriculationnumber_lastname_topic1_result.txt)

utterance_ID   predicted_dialog_act


* Please do not include the utterances in the submission file




