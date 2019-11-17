// Client side implementation of UDP client-server model 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
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
	char *buffer = malloc(sizeof(char)*MAXLINE); 
	struct sockaddr_in	 servaddr; 

	// Creating socket file descriptor 
               //todo
	if((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)  
    {
        printf("create udp sockfd");  
        exit(1);
    }
	bzero(&servaddr, sizeof(servaddr)); 

	memset(&servaddr, 0, sizeof(servaddr)); 
	
	// Filling server information 
	servaddr.sin_family = AF_INET; 
	servaddr.sin_port = htons(PORT); 
	servaddr.sin_addr.s_addr = INADDR_ANY; 
	
	int n, len; 
	printf("Please enter string, enter exit means close the process\n");
	while(1) {
		printf("Enter message:");

		//get user input
		//todo
		len = sizeof(servaddr);
		memset(buffer, 0, MAXLINE);
        gets(buffer);

		//send message
		//todo 
		n = sendto(sockfd, buffer, strlen(buffer), 0, (struct sockaddr*)&servaddr, len);
		if (n == -1)
        {
            printf("fail to send");
            exit(1);
        }

		if ((strncmp(buffer, "exit", 4)) == 0) { 
			printf("Client Exit...\n"); 
			break; 
		}

		//receive message
		//todo
		memset(buffer, 0, MAXLINE);
		n = recvfrom(sockfd, buffer, MAXLINE, 0, (struct sockaddr*)&servaddr, &len);
		if (n == -1)
		{
			printf("fail to receive");
			exit(1);
		}
		else
		{
			buffer[n] = '\0';
			printf("Received server : %s\n\n", buffer);
		}

	}

	close(sockfd); 
	return 0; 
} 
