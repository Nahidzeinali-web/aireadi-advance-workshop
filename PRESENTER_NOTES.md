# Presenter notes: AI-READI synthetic dataset demo

Generated from `build_notebook.py`; cell numbers match the notebook top to bottom. Minutes are rough speaking time. Cells marked optional can be skipped if running long.
Total if every cell is presented: about 44 minutes.


## Cell 1: AI-READI in the Cloud: a hands-on tour of the synthetic T2DM dataset (~1 min)

Welcome everyone. Today is about how quickly you can go from zero to real questions on a large multimodal
diabetes dataset, entirely in the cloud. No deep learning background is needed.

Frame the session: everything you see runs from one notebook against a read-only S3 bucket. Nothing is
downloaded in bulk; we pull only what each step needs.

Mention that attendees can follow along on their own VM if they have credentials, or just watch.

## Cell 2: 1. What is AI-READI? (~3 min)

Two or three minutes, no more. The points that matter for the rest of the demo:

1. It is deliberately balanced by diabetes severity and by race and ethnicity. That design is what makes
   fairness and "path back to health" questions answerable.
2. It is multimodal per person: eyes, heart, glucose, activity, environment, labs. Most diabetes datasets have
   one or two of these.
3. The retinal images come from six different devices. Remember this; it is the setup for the game in Part 5.
4. Everything is in a standard format (DICOM, WFDB, OMOP), so the same code works on the real release.

If asked about access: real data is on FAIRhub (fairhub.io) under a license; restricted fields such as
race/ethnicity, sex, zip code and medications need a separate request. Use is limited to T2DM research.

## Cell 3: 1.1 Why a synthetic version? (~2 min)

This is the most important caveat of the talk, so say it plainly: synthetic data is for exploring, teaching and
building pipelines. Findings have to be confirmed on the real data.

Explain the linkage table. Inside the tabular cohort, the generator produced whole participants, so labs, ECG,
wearable and CGM values for participant 42 belong together and their relationships were learned from real data.
Images were generated separately, one synthetic person per image, so you cannot link a retina to an HbA1c here.

Why ADVANCE: it is a Vanderbilt center (Biomedical Informatics) with deep privacy expertise; Brad Malin is a
co-director and an author on the synthetic dataset, together with Aaron Lee.

## Cell 4: 2. Connect to the data (~1 min)

While this cell is on screen, show where the credentials ZIP lives on the VM (or say you exported the three
environment variables before starting Jupyter). Point out the expiry: the instructions promise 12 hours but in our
testing the tokens died after about one hour. Download the ZIP within the hour before the session. If a cell
starts failing with ExpiredToken, download a fresh ZIP and rerun the credentials cell (the cache means nothing
already downloaded is lost).

## Cell 5: code, in 2. Connect to the data (~0.5 min)

Run it and move on. On a Deep Learning VM image everything is usually present already. If torch has to be
installed live it can take a few minutes, so do this before the session starts.

## Cell 6: code, in 2. Connect to the data (~0.5 min)

Configuration lives in one place. Mention the two identical buckets: ai-readi-sd in us-east-2 and
ai-readi-bucket in us-east-1. Pick the one in the same region as your VM; transfers inside a region are fast
and free.

The cache folder means every file is downloaded once. Rerunning the notebook is quick.

## Cell 7: code, in 2. Connect to the data (~0.5 min)

The output should say the role is s3-readonly. Point out that the notebook never prints the keys.

If this fails: ExpiredToken means the token has lapsed (about an hour in practice); AccessDenied usually means the variables from an older
ZIP are still exported in the shell. Restart the kernel after exporting new ones.

## Cell 8: 2.1 What is in the bucket (~1 min)

Walk the table top to bottom. Stress that the manifests and clinical CSVs are tiny (well under 300 MB together),
so you can explore metadata for the whole dataset before downloading a single image.

The note about 26,708 unlisted OCT files is a real quirk we found; mention it only if someone asks about file
counts.

## Cell 9: code, in 2.1 What is in the bucket (~0.5 min)

Live proof that we are reading from S3. Every data type has a clinical_data folder or a manifest next to the
files. That pattern mirrors the real AI-READI release.

## Cell 10: code, in 2.1 What is in the bucket (~0.5 min)

Two helpers carry the whole notebook: fetch one key into the cache, or fetch many in parallel threads.
The parallel version matters for images: S3 is fast per connection but has per-request latency, so 32 threads
is roughly 20 to 30 times faster than a plain loop.

Expect around 10 to 30 seconds in-region for about 250 MB.

## Cell 11: 3. A tour of the data types (~0.25 min)

Transition: we have the metadata for all 150,000 images on disk now. First question is always what was captured
and on which machine.

## Cell 12: code, in 3. A tour of the data types (~1 min)

This matrix is the design of the imaging arm: the same eye photographed by many devices. Point out that color
photos come from four devices, infrared from four, OCT from four, and only the Eidon does autofluorescence.
Blank cells mean that device does not capture that type.

For the real data this is what makes cross-device research possible: the same people, the same day, different
machines.

## Cell 13: code, in 3. A tour of the data types (~2 min)

Give the audience ten seconds to look. These are all synthetic, generated by a diffusion model, yet they carry
each device's look: the Aurora's small logo in the corner, the Eidon's text overlays and mosaic shape, the
Triton's saturated orange, the square Spectralis infrared field, and different OCT noise textures.

Ask the room: could you tell an OCT from the Triton and the Maestro2 apart? Both are Topcon. Park that thought
for Part 5.

Occasionally a sampled OCT image is almost pure noise; the generator is not perfect. That is also worth saying.

## Cell 14: code, in 3. A tour of the data types (~0.5 min, optional)

Real DICOM headers: the device, eye, image size and the standard SOP class are all in the file itself, not just
in the manifest. That means any DICOM tool (viewers, PACS, MONAI, pydicom) works on this data.

Data-quality footnote for the technical crowd: infrared and OCT headers say MONOCHROME1 (which would mean
"invert for display"), but the pixels are stored the normal way round. Do not blindly apply inversion.

## Cell 15: code, in 3. A tour of the data types (~1 min)

Every image comes with three disease labels in OMOP format. Prevalence is about 5 percent AMD, 8 percent glaucoma
and 4 percent diabetic retinopathy, and it is nearly identical across devices.

That flatness matters for Part 5: a device classifier cannot cheat by learning that one device sees sicker eyes.

## Cell 16: 3.2 ECG: machine measurements for every participant (~0.5 min)

Transition from the eye to the heart. From here on we are in the tabular cohort of 10,518 synthetic
participants, where every file shares the same participant_id.

Say the waveform caveat plainly: anyone who wants to train on raw ECG signals needs the real release. The
measurements are still useful, and Part 4 shows they line up with the wearable data.

## Cell 17: code, in 3.2 ECG: machine measurements for every participant (~1 min)

WFDB is the PhysioNet format, so the wfdb package reads a record in one line. The header carries the device, the
machine measurements and the device's own text interpretation (for example "Normal sinus rhythm" and an
abnormal flag). The last printed line confirms the flat waveform.

The histograms show the measurements across the whole cohort; the ranges look physiological (heart rate
roughly 40 to 105, QTc centred near 420 ms).

If someone asks: the manifest has a person_id column that is constant; use participant_id.

## Cell 18: 3.3 Wearable and CGM summaries, clinical labs and conditions (~0.5 min)

Be upfront that the synthetic wearable table is a summary. It is still useful: it is linked to the same synthetic
participants as the labs and ECG, which is what Part 4 needs.

## Cell 19: code, in 3.3 Wearable and CGM summaries, clinical labs and conditions (~1 min)

Three tables, one participant key. The OMOP tables are long format (one row per measurement), so we pivot to one
row per person, which is the shape most ML code wants.

Point at blood_glucose_mean and blood_glucose_std: those are the CGM features. heart_rate_mean, stress_mean,
steps and sleep hours come from the Garmin side.

## Cell 20: code, in 3.3 Wearable and CGM summaries, clinical labs and conditions (~0.5 min, optional)

The condition list reads like a real T2DM-risk cohort: high blood pressure, high cholesterol, elevated A1C,
arthritis, type 2 diabetes (about 38 percent), obesity and prediabetes lead the list. That is the balanced
recruitment design showing through.

## Cell 21: 4. Relationships across data types (~0.5 min)

Set up the logic: we will use known physiology as a test of the synthetic data. If the ECG heart rate and the
wearable heart rate are unrelated, the generator broke the link and we should not trust subtler findings.

## Cell 22: code, in 4. Relationships across data types (~2 min)

Let the heatmap sit for a moment, then walk three blocks (numbers are shown only where the correlation is at
least 0.2):

1. Glycemia block: HbA1c, serum glucose, CGM mean glucose and type 2 diabetes all move together. The CGM file and
   the lab file are different tables, so this link survived synthesis.
2. Heart block: ECG heart rate against wearable mean and resting heart rate is about 0.7. Two different devices,
   two different files, consistent physiology.
3. ECG internal: QT shortens as heart rate rises (negative), which is textbook.

Then point at the blanks: steps against BMI is near zero, and systolic blood pressure barely relates to the
high blood pressure diagnosis. Section 4.2 makes that explicit.

## Cell 23: code, in 4. Relationships across data types (~1 min)

This is the fidelity scorecard in miniature. The first five rows behave as expected. The last three (blood
pressure vs hypertension diagnosis, steps vs BMI, creatinine vs kidney problems) are much weaker than you would
see in real people.

Message: synthetic data preserved the strong, common relationships and flattened some weaker or rarer ones.
Use it to build and debug pipelines, then confirm on the real release. Measuring this gap systematically is
itself a research question (Part 6).

## Cell 24: code, in 4. Relationships across data types (~1 min)

Same story from the diagnosis side. People who report type 2 diabetes have higher HbA1c (about 6.4 vs 5.5
percent), higher CGM mean glucose and higher BMI. Three different tables, one consistent picture.

Good moment to ask the room what they would do with a real version of this: for example, which CGM patterns
separate insulin users from people controlled by lifestyle alone.

## Cell 25: code, in 4. Relationships across data types (~0.5 min, optional)

One scatter to make the cross-device link concrete: a 10 second clinic ECG against days of wrist-worn heart rate.
Optional if short on time.

## Cell 26: 5. Challenge: which device took this scan? (~2 min)

This is the main event. Credit Aaron Lee's demo from last year: people were shown grayscale random crops of
retinal images and asked to name the device; they could not, the classifier could.

Why it matters: if a model can tell the device from a tiny patch, then any disease model trained
on images can silently use the device as a proxy (for example if one clinic with sicker patients uses one
machine). That is the domain shift problem that holds back deployment.

Twist for the synthetic version: the generator was told the device for every image. If a classifier recovers the
device, the generator reproduced the fingerprint, which is a nontrivial test of how realistic the images are.

## Cell 27: code, in 5. Challenge: which device took this scan? (~1 min)

Pulls a balanced sample, 1,000 B-scans per device, directly from S3 with 32 threads. In-region this takes
well under a minute. Color photos get converted to grayscale, which was the original trick for removing the
obvious color cues.

Every image belongs to a different synthetic participant, so a random train/test split has no patient leakage.

## Cell 28: code, in 5. Challenge: which device took this scan? (~1 min)

Walk left to right. With the whole scan you can see layer shape and background. A 64 pixel crop keeps some
texture. Once brightness and contrast are removed, most people are guessing. At 32 pixels there is almost
nothing a human can use.

## Cell 29: 5.1 Your turn (~0.5 min)

Give the room about a minute. If people are following along, they type their answer string into the next cell;
otherwise ask for a show of hands on two or three crops.

## Cell 30: code, in 5.1 Your turn (~1.5 min)

The top row is the "training data" for humans: two labeled crops per device. The bottom two rows are the quiz.
The answer key is revealed in cell 5.3 after the model has trained, so do not scroll ahead.

## Cell 31: code, in 5.1 Your turn (~0.5 min)

Hands-on attendees fill this in. If nobody does, it is scored as an empty answer and the reveal still works.

## Cell 32: 5.2 Now the model (~0.5 min)

While the next cell runs, talk through what the network is: about 100 thousand parameters, far smaller than any
modern vision model, no pretraining. A model this small doing this well tells you the fingerprint is easy to
learn.

## Cell 33: code, in 5.2 Now the model (~3 min)

Expected results from our dry run on OCT (GPU, 12 epochs): whole about 99 percent, crop64 about 92, crop64n
about 90, crop32n about 80. On CPU with 6 epochs the harder levels land a few points lower. Chance is 25.

If time is short, set LEVELS to ["whole", "crop64n"] before running; the story still works.

## Cell 34: code, in 5.2 Now the model (~1 min)

The headline chart. Even a 32 pixel patch with brightness and contrast removed is identified far above chance.
The model is reading texture: speckle and noise structure, not anatomy.

## Cell 35: code, in 5.2 Now the model (~1 min)

Look for the biggest off-diagonal cells. In our runs the Maestro2 and the Triton are confused most often, and
they are both Topcon devices. Same manufacturer, similar processing pipeline, similar fingerprint. Spectralis is
the easiest to spot.

## Cell 36: 5.3 The reveal (~0.25 min)

Ask everyone to lock in their answers before running the next cell.

## Cell 37: code, in 5.3 The reveal (~1.5 min)

Read out the model's score, then ask who beat it. Almost nobody does. That is the point: the fingerprint is real
(here, reproduced by the generator), invisible to us, and trivially learnable by a machine.

## Cell 38: code, in 5.3 The reveal (~1.5 min, optional)

The explanation slide. After removing brightness and contrast, the remaining difference is how much fine-grained
texture each device has. Spectralis images are smoother (power drops fastest at high frequency), Maestro2 is the
grainiest. A human eye does not measure this; a convolution does it in its first layer.

Optional: rerun Part 5 with MODALITY = "Color Photography". Grayscale color photos are about 98 percent
identifiable from the whole image and still well above chance from crops.

## Cell 39: 6. Problems worth a grant (~4 min)

Pick two to discuss depending on the audience. For an imaging audience lead with 1 and 2; for a clinical
audience lead with 3 and 4; for informatics and privacy people lead with 5.

Why these are strong grant angles: each uses something rare about AI-READI (multi-device imaging of the same
people, per-person multimodality, deliberate balance across severity and race and ethnicity), and each has a
synthetic-data prototype path, which lowers risk in a proposal.

Invite the room: what would you add?

## Cell 40: 7. Next steps and resources (~1 min)

Close by repeating the two caveats (synthetic images are not linked to the tabular participants; confirm
findings on real data) and the one invitation (pick a problem from Part 6 and start prototyping this week).
