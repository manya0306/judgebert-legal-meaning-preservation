# Error analysis - lexrule_gbm (out-of-fold predictions, repeat 0, grouped CV)

All rows are real dataset rows (AI-generated or original synthetic); predictions are genuine out-of-fold outputs. Labels are AI-assisted, not human.

## Obligation changed  (n=43)

mean predicted 3.76 vs mean reference 2.72; MAE 1.27

- `AIG-C001-HI-06` (hi, tenancy) ref 1 / pred 7.0
  - O: मकान मालिक को किराये के परिसर में प्रवेश करने से पहले किरायेदार को कम से कम 24 घंटे पूर्व सूचना देनी होगी, आपातकाल की स्थिति को छोड़कर।
  - S: मकान मालिक किसी भी समय किराये के परिसर में प्रवेश कर सकता है और बाद में किरायेदार को बता सकता है।
- `AIG-C001-EN-06` (en, tenancy) ref 1 / pred 6.9
  - O: The landlord shall give the tenant at least 24 hours' prior notice before entering the rented premises, except in an emergency.
  - S: The landlord may enter the rented premises at any time and may inform the tenant afterwards.
- `SYN-EN-001` (en, tenancy) ref 2 / pred 7.4
  - O: The tenant shall give 30 days' written notice before vacating.
  - S: The tenant may tell the landlord before leaving.

## Permission/obligation swapped  (n=31)

mean predicted 3.85 vs mean reference 2.42; MAE 1.89

- `SYN-HI-006` (hi, tenancy) ref 1 / pred 7.4
  - O: किरायेदार मकान में किसी भी प्रकार का स्थायी बदलाव नहीं करेगा।
  - S: किरायेदार मकान में कुछ बदलाव कर सकता है।
- `AIG-C002-HI-05` (hi, tenancy) ref 2 / pred 7.8
  - O: किरायेदार मकान मालिक की पूर्व लिखित सहमति के बिना परिसर का कोई भी भाग उप-किराये पर नहीं देगा।
  - S: किरायेदार मकान मालिक को लिखित सूचना देने के बाद परिसर का कोई भी भाग उप-किराये पर दे सकता है।
- `AIG-C007-EN-05` (en, employment) ref 1 / pred 6.0
  - O: The employer shall not deduct any amount from an employee's wages except for tax, provident fund contributions, or amounts authorised by the employee in writing.
  - S: The employer may deduct amounts from an employee's wages for tax, provident fund contributions, or any other purpose it considers necessary.

## Contradiction / negation / reversal  (n=44)

mean predicted 4.36 vs mean reference 1.34; MAE 3.02

- `AIG-C018-HI-05` (hi, contract) ref 1 / pred 8.6
  - O: क्रेता मानक-प्रारूप संविदा की ऐसी किसी शर्त से बाध्य नहीं होगा जो संविदा पर हस्ताक्षर होने से पहले क्रेता के ध्यान में नहीं लाई गई थी।
  - S: क्रेता मानक-प्रारूप संविदा की हर शर्त से बाध्य होगा, चाहे वह क्रेता के ध्यान में लाई गई हो या नहीं।
- `AIG-C018-EN-05` (en, contract) ref 1 / pred 7.8
  - O: The buyer shall not be bound by any term of the standard-form contract that was not brought to the buyer's notice before the contract was signed.
  - S: The buyer shall be bound by every term of the standard-form contract, whether or not it was brought to the buyer's notice.
- `AIG-C031-EN-05` (en, administrative) ref 1 / pred 7.8
  - O: No licence shall be cancelled unless the licence holder has been given a show-cause notice and a reasonable opportunity to reply.
  - S: A licence may be cancelled at any time, and the holder shall be informed afterwards.

## Exception removed  (n=48)

mean predicted 4.09 vs mean reference 3.52; MAE 1.37

- `AIG-C001-HI-06` (hi, tenancy) ref 1 / pred 7.0
  - O: मकान मालिक को किराये के परिसर में प्रवेश करने से पहले किरायेदार को कम से कम 24 घंटे पूर्व सूचना देनी होगी, आपातकाल की स्थिति को छोड़कर।
  - S: मकान मालिक किसी भी समय किराये के परिसर में प्रवेश कर सकता है और बाद में किरायेदार को बता सकता है।
- `AIG-C001-EN-06` (en, tenancy) ref 1 / pred 6.9
  - O: The landlord shall give the tenant at least 24 hours' prior notice before entering the rented premises, except in an emergency.
  - S: The landlord may enter the rented premises at any time and may inform the tenant afterwards.
- `AIG-C014-HI-03` (hi, consumer) ref 5 / pred 1.0
  - O: शिकायत उस तिथि से दो वर्ष के भीतर दाखिल की जाएगी जिस दिन वाद का कारण उत्पन्न हुआ, और फोरम विलंबित शिकायत केवल लिखित रूप में दर्ज पर्याप्त कारण होने पर ही स्वीकार कर सकता है।
  - S: शिकायत उस तिथि से दो वर्ष के भीतर दाखिल की जाएगी जिस दिन वाद का कारण उत्पन्न हुआ।

## Condition altered  (n=52)

mean predicted 4.00 vs mean reference 3.07; MAE 1.50

- `AIG-C031-EN-05` (en, administrative) ref 1 / pred 7.8
  - O: No licence shall be cancelled unless the licence holder has been given a show-cause notice and a reasonable opportunity to reply.
  - S: A licence may be cancelled at any time, and the holder shall be informed afterwards.
- `AIG-C002-HI-05` (hi, tenancy) ref 2 / pred 7.8
  - O: किरायेदार मकान मालिक की पूर्व लिखित सहमति के बिना परिसर का कोई भी भाग उप-किराये पर नहीं देगा।
  - S: किरायेदार मकान मालिक को लिखित सूचना देने के बाद परिसर का कोई भी भाग उप-किराये पर दे सकता है।
- `AIG-C009-HI-05` (hi, insurance) ref 1 / pred 4.6
  - O: बीमाकर्ता किसी दावे को गैर-प्रकटीकरण के आधार पर केवल तभी अस्वीकार कर सकता है जब छिपाया गया तथ्य महत्वपूर्ण रहा हो और पॉलिसी जारी करते समय बीमाकर्ता को ज्ञात न रहा हो।
  - S: बीमाकर्ता किसी भी तथ्य के गैर-प्रकटीकरण के आधार पर दावा अस्वीकार कर सकता है, चाहे वह महत्वपूर्ण हो या नहीं।

## Actor / party changed  (n=35)

mean predicted 4.21 vs mean reference 2.89; MAE 1.84

- `AIG-C040-HI-06` (hi, motor_vehicles) ref 1 / pred 7.6
  - O: चोट पहुँचाने वाली दुर्घटना में शामिल वाहन का चालक रुकेगा, घायल व्यक्ति की सहायता करेगा, और चौबीस घंटे के भीतर निकटतम थाने में दुर्घटना की सूचना देगा।
  - S: चोट पहुँचाने वाली दुर्घटना में शामिल वाहन का चालक वहाँ से चला जाएगा और स्वामी को दुर्घटना की सूचना देगा।
- `AIG-C024-HI-06` (hi, property) ref 1 / pred 4.7
  - O: जिस सह-स्वामी ने संपत्ति कर में अपने हिस्से से अधिक का भुगतान किया है, वह अधिक राशि अन्य सह-स्वामियों से उनके हिस्सों के अनुपात में वसूल कर सकता है।
  - S: जिस सह-स्वामी ने संपत्ति कर में अपने हिस्से से अधिक का भुगतान किया है, वह अधिक राशि अन्य सह-स्वामियों को उनके हिस्सों के अनुपात में देगा।
- `AIG-C037-HI-01` (hi, privacy) ref 8 / pred 4.4
  - O: डेटा न्यासी व्यक्तिगत डेटा का प्रसंस्करण केवल उसी प्रयोजन के लिए करेगा जिसके लिए डेटा प्रधान ने सहमति दी है, और वह प्रयोजन पूरा हो जाने पर डेटा मिटा देगा।
  - S: व्यक्तिगत डेटा संभालने वाली कंपनी उसे केवल उसी प्रयोजन के लिए उपयोग कर सकती है जिसके लिए व्यक्ति ने हामी भरी हो, और प्रयोजन पूरा होने पर उसे मिटाना होगा।

## Scope / quantifier changed  (n=59)

mean predicted 4.25 vs mean reference 3.64; MAE 1.52

- `AIG-C007-EN-05` (en, employment) ref 1 / pred 6.0
  - O: The employer shall not deduct any amount from an employee's wages except for tax, provident fund contributions, or amounts authorised by the employee in writing.
  - S: The employer may deduct amounts from an employee's wages for tax, provident fund contributions, or any other purpose it considers necessary.
- `AIG-C007-HI-05` (hi, employment) ref 1 / pred 5.1
  - O: नियोक्ता कर्मचारी के वेतन से कर, भविष्य निधि अंशदान, या कर्मचारी द्वारा लिखित रूप में अधिकृत राशि के सिवाय कोई राशि नहीं काटेगा।
  - S: नियोक्ता कर, भविष्य निधि अंशदान, या अन्य किसी आवश्यक प्रयोजन के लिए कर्मचारी के वेतन से राशि काट सकता है।
- `AIG-C004-HI-06` (hi, tenancy) ref 1 / pred 5.0
  - O: सुरक्षा जमा राशि दो महीने के किराये से अधिक नहीं होगी और किरायेदार के घर खाली करने के तीस दिन के भीतर लौटाई जाएगी, केवल बकाया किराये और सामान्य टूट-फूट से अधिक क्षति की लागत काटकर।
  - S: मकान मालिक परिसर को हुई किसी भी क्षति के मुआवज़े के रूप में सुरक्षा जमा राशि रख सकता है।

## Time limit changed  (n=34)

mean predicted 4.12 vs mean reference 3.71; MAE 1.06

- `AIG-C004-HI-06` (hi, tenancy) ref 1 / pred 5.0
  - O: सुरक्षा जमा राशि दो महीने के किराये से अधिक नहीं होगी और किरायेदार के घर खाली करने के तीस दिन के भीतर लौटाई जाएगी, केवल बकाया किराये और सामान्य टूट-फूट से अधिक क्षति की लागत काटकर।
  - S: मकान मालिक परिसर को हुई किसी भी क्षति के मुआवज़े के रूप में सुरक्षा जमा राशि रख सकता है।
- `AIG-C041-EN-06` (en, intellectual_property) ref 1 / pred 4.3
  - O: The copyright in a literary work shall subsist until sixty years after the death of the author, and in the case of a work of joint authorship, until sixty years after the death of the author who dies last.
  - S: The copyright in a literary work shall subsist until sixty years before the death of the author.
- `AIG-C014-HI-06` (hi, consumer) ref 3 / pred 5.4
  - O: शिकायत उस तिथि से दो वर्ष के भीतर दाखिल की जाएगी जिस दिन वाद का कारण उत्पन्न हुआ, और फोरम विलंबित शिकायत केवल लिखित रूप में दर्ज पर्याप्त कारण होने पर ही स्वीकार कर सकता है।
  - S: शिकायत निर्णय की तिथि से दो वर्ष के भीतर दाखिल की जाएगी, और फोरम विलंबित शिकायत केवल लिखित रूप में दर्ज पर्याप्त कारण होने पर ही स्वीकार कर सकता है।

## Numerical change  (n=103)

mean predicted 4.12 vs mean reference 3.80; MAE 1.05

- `AIG-C001-HI-06` (hi, tenancy) ref 1 / pred 7.0
  - O: मकान मालिक को किराये के परिसर में प्रवेश करने से पहले किरायेदार को कम से कम 24 घंटे पूर्व सूचना देनी होगी, आपातकाल की स्थिति को छोड़कर।
  - S: मकान मालिक किसी भी समय किराये के परिसर में प्रवेश कर सकता है और बाद में किरायेदार को बता सकता है।
- `AIG-C001-EN-06` (en, tenancy) ref 1 / pred 6.9
  - O: The landlord shall give the tenant at least 24 hours' prior notice before entering the rented premises, except in an emergency.
  - S: The landlord may enter the rented premises at any time and may inform the tenant afterwards.
- `SYN-EN-001` (en, tenancy) ref 2 / pred 7.4
  - O: The tenant shall give 30 days' written notice before vacating.
  - S: The tenant may tell the landlord before leaving.

## Reference changed  (n=2)

mean predicted 3.11 vs mean reference 3.00; MAE 0.11

- `AIG-C030-EN-04` (en, administrative) ref 3 / pred 3.1
  - O: An order passed under section 14 shall not take effect until the affected person has been served a copy and given fifteen days to file an appeal.
  - S: An order passed under section 41 shall not take effect until the affected person has been served a copy and given fifteen days to file an appeal.
- `AIG-C030-HI-04` (hi, administrative) ref 3 / pred 3.1
  - O: धारा 14 के अंतर्गत पारित आदेश तब तक प्रभावी नहीं होगा जब तक प्रभावित व्यक्ति को उसकी प्रति तामील न कर दी जाए और अपील दाखिल करने के लिए पंद्रह दिन न दे दिए जाएँ।
  - S: धारा 41 के अंतर्गत पारित आदेश तब तक प्रभावी नहीं होगा जब तक प्रभावित व्यक्ति को उसकी प्रति तामील न कर दी जाए और अपील दाखिल करने के लिए पंद्रह दिन न दे दिए जाएँ।

## HIGH lexical similarity (Jaccard>=0.8) but NOT preserved  (n=113)

mean predicted 3.22 vs mean reference 2.57; MAE 0.85

- `SYN-EN-019` (en, insurance) ref 1 / pred 5.2
  - O: Damage caused by flood is not covered under this policy.
  - S: Damage caused by flood is covered under this policy.
- `SYN-HI-019` (hi, insurance) ref 1 / pred 5.1
  - O: बाढ़ से हुई क्षति इस पॉलिसी के अंतर्गत कवर नहीं है।
  - S: बाढ़ से हुई क्षति इस पॉलिसी के अंतर्गत कवर है।
- `AIG-C024-HI-06` (hi, property) ref 1 / pred 4.7
  - O: जिस सह-स्वामी ने संपत्ति कर में अपने हिस्से से अधिक का भुगतान किया है, वह अधिक राशि अन्य सह-स्वामियों से उनके हिस्सों के अनुपात में वसूल कर सकता है।
  - S: जिस सह-स्वामी ने संपत्ति कर में अपने हिस्से से अधिक का भुगतान किया है, वह अधिक राशि अन्य सह-स्वामियों को उनके हिस्सों के अनुपात में देगा।

## LOW lexical similarity (Jaccard<0.5) but Preserved  (n=116)

mean predicted 6.98 vs mean reference 8.48; MAE 1.58

- `AIG-C042-EN-01` (en, environment) ref 9 / pred 2.4
  - O: No industry shall discharge effluents into a river unless the effluents have been treated to the prescribed standards and the industry holds a valid consent from the pollution control board.
  - S: Factories may release waste water into a river only after treating it to the set standards and holding valid consent from the pollution control board.
- `AIG-C021-EN-01` (en, banking) ref 9 / pred 3.3
  - O: A bank shall not disclose information about a customer's account to any third party except where the customer has consented in writing or the disclosure is required by an order of a court.
  - S: Banks may share a customer's account details with outsiders only with the customer's written consent or when a court orders it.
- `AIG-C032-EN-01` (en, criminal_procedure) ref 9 / pred 3.4
  - O: A police officer shall not arrest a person for an offence punishable with imprisonment of less than seven years unless the officer records in writing the reasons why the arrest is necessary.
  - S: For an offence carrying under seven years' imprisonment, police may arrest only after writing down why the arrest is needed.

## Largest errors, language=hi  (n=276)

mean predicted 4.83 vs mean reference 5.10; MAE 1.52

- `AIG-C018-HI-05` (hi, contract) ref 1 / pred 8.6
  - O: क्रेता मानक-प्रारूप संविदा की ऐसी किसी शर्त से बाध्य नहीं होगा जो संविदा पर हस्ताक्षर होने से पहले क्रेता के ध्यान में नहीं लाई गई थी।
  - S: क्रेता मानक-प्रारूप संविदा की हर शर्त से बाध्य होगा, चाहे वह क्रेता के ध्यान में लाई गई हो या नहीं।
- `AIG-C040-HI-06` (hi, motor_vehicles) ref 1 / pred 7.6
  - O: चोट पहुँचाने वाली दुर्घटना में शामिल वाहन का चालक रुकेगा, घायल व्यक्ति की सहायता करेगा, और चौबीस घंटे के भीतर निकटतम थाने में दुर्घटना की सूचना देगा।
  - S: चोट पहुँचाने वाली दुर्घटना में शामिल वाहन का चालक वहाँ से चला जाएगा और स्वामी को दुर्घटना की सूचना देगा।
- `AIG-C023-HI-01` (hi, property) ref 9 / pred 2.5
  - O: कोई भी व्यक्ति नगरपालिका प्राधिकारी की स्वीकृति के बिना भूमि पर कोई इमारत नहीं बनाएगा, और बिना स्वीकृति के किया गया कोई भी निर्माण तीस दिन का नोटिस देने के बाद ध्वस्त किया जा सकता है।
  - S: भूमि पर निर्माण के लिए नगरपालिका प्राधिकारी की मंज़ूरी चाहिए; बिना मंज़ूरी के बना कोई भी निर्माण तीस दिन का नोटिस देने के बाद गिराया जा सकता है।

## Largest errors, language=en  (n=276)

mean predicted 4.89 vs mean reference 5.11; MAE 1.43

- `AIG-C018-EN-05` (en, contract) ref 1 / pred 7.8
  - O: The buyer shall not be bound by any term of the standard-form contract that was not brought to the buyer's notice before the contract was signed.
  - S: The buyer shall be bound by every term of the standard-form contract, whether or not it was brought to the buyer's notice.
- `AIG-C031-EN-05` (en, administrative) ref 1 / pred 7.8
  - O: No licence shall be cancelled unless the licence holder has been given a show-cause notice and a reasonable opportunity to reply.
  - S: A licence may be cancelled at any time, and the holder shall be informed afterwards.
- `AIG-C035-EN-01` (en, company) ref 9 / pred 2.3
  - O: A director shall disclose to the board any personal interest in a proposed contract before the board discusses it, and shall not vote on it.
  - S: Before the board discusses a proposed contract, a director must tell it about any personal interest in the contract, and may not vote on it.
