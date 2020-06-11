# BECLoMA cl appendix

This is the online appendix of (BECLoMA)[https://ieeexplore.ieee.org/document/8330252].
It shares an executable version of the tool that can be run with the following command:

```
java -jar becloma_cl.jar <path source code><path crash log><path user reviews>
```

The first argument is the root for the project for the mobile app you want to use. 
The second is the path to the `txt` file containing the crash log. It is worth note that the crash log might have the same structure of the logs in output from either (Monkey)[https://developer.android.com/studio/test/monkey] or (Sapienz)[https://github.com/Rhapsod/sapienz].
The third argument is the path of the `csv` file containing the user reviews.

You can find examples of correctly formatted input files in (this)[https://github.com/sealuzh/becloma-info/tree/master/test-files] folder.

You can try to execute the following command to perform the link.

```
java -jar becloma_cl.jar test-files/com.amaze.filemanager.gz test-files/crash-log.txt test-files/com.amaze.filemanager.csv
```

You will have a `txt` file in output with the detected (if any) links and the indication of the crash log and the review linked together. 

## Classification of user reviews

BECLoMA aims at linking user reviews with crash logs. 
However, to do so, we first need to select the reviews actually discussing feature bugs and crashes. 
We built a classifier trained on manually labelled reviews that is able to automatically perform such a classification.
We share its entire code [here](https://github.com/sealuzh/becloma-info/tree/master/classifier)
