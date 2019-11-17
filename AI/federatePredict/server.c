// Server side implementation of UDP client-server model 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>
#include <ctype.h>
#include <string.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 

#define PORT	 8080 
#define MAXLINE 1024 

// Driver code 
int main() { 
	int sockfd; 
	char buffer[MAXLINE]; 
	struct sockaddr_in servaddr, cliaddr; 
	
	// Creating socket file descriptor 
                //todo
	if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)  
    {
        perror("create udp sockfd");  
        exit(1);
    }
	bzero(&servaddr, sizeof(servaddr));  


	memset(&servaddr, 0, sizeof(servaddr)); 
	memset(&cliaddr, 0, sizeof(cliaddr)); 
	
	// Filling server information 
	servaddr.sin_family = AF_INET; // IPv4 
	servaddr.sin_addr.s_addr = INADDR_ANY; 
	servaddr.sin_port = htons(PORT); 
	
	// Bind the socket with the server address 
                //todo
	 if(bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)  
    {
        perror("Bind");  
        exit(1);  
    }
	
	int len, n; 
	while(1) {

	//receive message
	//todo
	len = sizeof(cliaddr);
	memset(buffer, 0, MAXLINE);
    n = recvfrom(sockfd, buffer, MAXLINE, 0, (struct sockaddr *)&cliaddr, &len);

    if(n < 0)
	{
		perror("fail receive");
		continue;
	}
	else
	{
		buffer[n] = '\0';
		if(strcmp(buffer, "exit") == 0)
		{
			break;
		}
		printf("Received client : %s\n", buffer);
		n = 0;
	}
	//convert to uppercase
	//todo
	char result[MAXLINE];
	for (int i = 0; i < strlen(buffer); i++)
	{
		result[i] = toupper(buffer[i]);
	}

	//send messgae
	//todo
	n = sendto(sockfd, result, strlen(buffer), 0, (struct sockaddr*)&cliaddr, len);
	if (n == -1)
	{
		printf("fail to reply");
		exit(1);
	}

	}
	if(close(sockfd) == -1)
    {
        printf("close sockfd");
        exit(1);
    }
	 
	return 0; 
} 
